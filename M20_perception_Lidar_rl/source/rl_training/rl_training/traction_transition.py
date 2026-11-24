"""Training-only traction/transition math; no simulator or deployment inputs."""
import math
import torch


def transition_speed(elapsed, profile):
    """Idle, partial drive, abrupt full drive, release and retained .6s ramp.

    Profiles 0..2 hold .3/.5/.7. Profile 3 steps 0->1->.5;
    profile 4 steps 0->.3->1; profile 5 retains the existing ramp to 1.
    """
    t = elapsed
    partial = t.new_tensor([.3, .5, .7, 1., .3, 1.])[profile]
    speed = torch.where(t >= 2., partial, 0.)
    speed = torch.where((profile == 3) & (t >= 6.), .5, speed)
    speed = torch.where((profile == 4) & (t >= 5.), 1., speed)
    return torch.where(profile == 5, ((t-2.)/.6).clamp(0, 1), speed)


def horizontal_patch(bottom, hits, normals, force_z, vertical_speed,
                     plane_tolerance=.025):
    """Conservative horizontal tread interior approximation, not a force normal.

    Five terrain rays (center and +/-3cm along world x/y) must agree on
    a horizontal plane. Reject edges, slopes, airborne wheels and landings.
    """
    finite = torch.isfinite(hits).all((-1, -2)) & torch.isfinite(normals).all((-1, -2))
    heights = torch.nan_to_num(hits[..., 2], nan=1e6, posinf=1e6, neginf=-1e6)
    level = (heights.amax(-1)-heights.amin(-1) < .008)
    horizontal = (normals[..., 2].abs() >= .98).all(-1)
    return (finite & level & horizontal & ((bottom-heights[..., 0]).abs() <= plane_tolerance)
            & (force_z >= 15.) & (vertical_speed.abs() <= .15))


def rolling_residual(linear, angular, radius):
    """Link-origin velocity + omega cross bottom offset, in world coordinates.

    Angular velocity already contains wheel spin; adding joint speed again
    double counts it. Only horizontal planes pass horizontal_patch().
    """
    offset = torch.zeros_like(linear)
    offset[..., 2] = -radius
    return (linear+torch.linalg.cross(angular, offset, dim=-1))[..., :2]


def traction_terms(residual, supported, speed, command):
    slip = torch.linalg.vector_norm(residual, dim=-1)
    weights = slip.new_tensor([.5, .5, 1., 1.])
    quality = 1./(1.+(slip/.25).square())
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)*(command > .1)
    # Fixed denominator: lifting/unloading a wheel cannot improve the reward.
    # A small progress-and-load credit makes the combined .45 reward / .10
    # cost prefer supported progress even when slip is very large. Both terms
    # vanish at standstill. Removing a wheel cannot improve their combined
    # value at the same actual progress; airborne swings receive no cost.
    reward = progress*((.25+.75*quality)*supported*weights).sum(1)/weights.sum()
    cost = progress*((1-quality)*supported*weights).sum(1)/weights.sum()
    return reward, cost, slip


def flat_pitch_cost(gravity, pitch_rate):
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=1))
    tilt = ((pitch.abs()-math.radians(8)).clamp_min(0)/math.radians(12)).square()
    rocking = ((pitch_rate.abs()-.65).clamp_min(0)/1.).square()
    return .7*tilt/(1+tilt)+.3*rocking/(1+rocking)


def transition_action_cost(action, previous_action, command, previous_command,
                           *, straight, entry_near):
    """Penalize command-step action impulses while preserving steady gait.

    The wheel action sign is deliberately not prescribed: the M20 wheel joint
    axis convention makes a raw action sign a poor proxy for forward motion.
    We constrain the *change* in the four wheel targets and sagittal front-leg
    targets only when a positive command appears or B1 is being approached.
    """
    rising = (command-previous_command > .08) & (command > .1) & (previous_command < .1)
    active = straight & (rising | entry_near)
    wheel_delta = action[:, 12:]-previous_action[:, 12:]
    leg_delta = action[:, [1, 2, 4, 5]]-previous_action[:, [1, 2, 4, 5]]
    wheel = (wheel_delta/.45).square().mean(1).clamp_max(4.)
    leg = (leg_delta/.55).square().mean(1).clamp_max(4.)
    return (wheel+.35*leg)*active


def entry_order_step(previous_support, support, previous_air, air, active,
                     age, first_side, first_kind, latched, *, confirmation_frames=2):
    """Detect simultaneous B1 lift/landing, without prescribing FL or FR.

    A single wheel gets a short lead window. If its mate lifts or settles in
    the same control tick or within two policy ticks, the event is latched as
    a jump-like bilateral transition. The first side remains symmetric.
    """
    support_onset = support & ~previous_support
    lift_onset = air & ~previous_air
    candidate = active & ~latched
    both = candidate & (support_onset.all(1) | lift_onset.all(1))
    single_support = candidate & (support_onset.sum(1) == 1)
    single_lift = candidate & (lift_onset.sum(1) == 1)
    any_single = single_support | single_lift
    kind = torch.where(single_support, torch.zeros_like(first_side), torch.ones_like(first_side))
    side = torch.where(any_single, torch.where(single_support[:, None], support_onset, lift_onset).long().argmax(1), first_side)
    old_age = age
    age = torch.where(any_single, torch.zeros_like(age), age)
    seen = (first_side >= 0) | any_single
    mate = torch.where(first_side.clamp_min(0) == 0, torch.ones_like(first_side), torch.zeros_like(first_side))
    matching_onset = torch.where(first_kind[:, None] == 0, support_onset, lift_onset)
    second = candidate & (first_side >= 0) & (first_kind >= 0) \
        & matching_onset.gather(1, mate[:, None]).squeeze(1) \
        & (old_age <= int(confirmation_frames))
    simultaneous = both | second
    latched = latched | simultaneous
    age = torch.where(seen & ~simultaneous, age+1, age)
    first_side = torch.where(any_single & (first_side < 0), side, first_side)
    first_kind = torch.where(any_single & (first_kind < 0), kind, first_kind)
    first_side = torch.where(simultaneous, torch.full_like(first_side, -2), first_side)
    first_kind = torch.where(simultaneous, torch.full_like(first_kind, -1), first_kind)
    return dict(previous_support=support, previous_air=air, age=age,
                first_side=first_side, first_kind=first_kind, latched=latched,
                simultaneous=simultaneous, lead=any_single & ~simultaneous)


def drive_band_loss(predicted, target, command):
    """Soft moving-teacher band permits less wheel spin, preserves nonzero drive.

    The band is action-space regularization, never an inferred traction label.
    Contact and progress rewards decide which speed within the band works.
    """
    teacher = target[:, 12:].detach().mean(1)
    drive = predicted[:, 12:].mean(1)*teacher.sign()
    scale = teacher.abs()
    error = (.65*scale-drive-.05).clamp_min(0).square()
    error += (drive-1.15*scale-.05).clamp_min(0).square()
    moving = command > .1
    return (error*moving).sum()/moving.sum().clamp_min(1)


def support_legs_loss(predicted, target):
    indices = [1, 2, 4, 5, 7, 8, 10, 11]
    error = ((predicted[:, indices]-target[:, indices].detach()).abs()-.05).clamp_min(0).square()
    return (error*error.new_tensor([1., 1., 1., 1., .5, .5, .5, .5])).sum(1).mean()/6.
