"""Flat front-led turning objectives in physical units; no action overrides."""
import math
import torch

PROTOCOL = 'm20_front_led_grounded_turn_v1'


def bounded(x):
    return x.square()/(1+x.square())


def turn_profile(elapsed, yaw, forward):
    """Raw joystick onset, stop and reversal; parent advances elapsed once."""
    cycle = (elapsed-1.).clamp_min(0).remainder(8.)
    moving = (elapsed >= 1.) & ((cycle < 3.) | ((cycle >= 4.) & (cycle < 7.)))
    direction = torch.where(cycle < 4., 1., -1.)
    return forward*moving, yaw*direction*moving


def front_follow_step(ready_time, front_started, released, force, height,
                      front_yaw_moment, onset, active, dt):
    """Rear reposition follows a loaded front initiation, never a timed kick.

    Rolling on all four wheels is valid. A front initiation can be a small
    supported steering moment, or an actual front reposition. Neither front
    wheel is required to lift; rear stepping is never rewarded by itself.
    """
    clear = onset | ~active
    started = torch.where(clear, False, front_started)
    front_air = ((height[:, :2] > .012) & (force[:, :2] < 15.)).any(1)
    started |= active & (front_air | (front_yaw_moment > 1.))
    front_loaded = (force[:, :2] >= 15.).all(1)
    ready_time = torch.where(clear | ~front_loaded, 0., ready_time+dt*started)
    ready = ready_time >= .08-1e-6
    rear_air = (height[:, 2:] > .012) & (force[:, 2:] < 15.)
    released = torch.where(clear[:, None], False, released)
    released |= rear_air
    completed = released.any(1) & (force[:, 2:] >= 15.).all(1)
    return dict(ready_time=torch.where(completed, 0., ready_time),
        front_started=started & ~completed, rear_released=released & ~completed[:, None],
        rear_unled=(rear_air & ~ready[:, None]).float().mean(1))


def turn_terms(command, velocity, yaw_rate, gravity, clearance, home_height,
               force, height, relative_velocity, reach, rear_unled,
               front_moment, rear_moment, active):
    yaw = command[:, 2]
    large = ((yaw.abs()-.30)/.25).clamp(0, 1)
    progress = (yaw_rate*yaw.sign()/yaw.abs().clamp_min(.1)).clamp(0, 1)
    tracking = torch.exp(-((yaw_rate-yaw)/.25).square())
    # Correct roll sign: a left lean has positive projected gravity y.
    lean = yaw.sign()*math.radians(3.)*(yaw.abs()/.55).clamp(0, 1)
    desired_height = home_height-(.012+.008*large)
    pose = (.45*bounded((gravity[:, 1]-lean.sin())/.04)
            +.30*bounded((clearance-desired_height)/.025)
            +.25*bounded((gravity[:, 0].abs()-.035).clamp_min(0)/.07))
    pure = command[:, :2].norm(dim=1) < .1
    drift = bounded(((velocity[:, :2]-command[:, :2]).norm(dim=1)-.025).clamp_min(0)/.10)
    backward = bounded((-velocity[:, 0]-.015).clamp_min(0)/.08)*(command[:, 0] >= 0)
    # For small turns, keep wheels loaded. Large turns permit front-first
    # reposition, with no prescribed diagonal gait or fixed leading side.
    missing = (force < 15.).float()
    small_air = 2.*(missing*missing.new_tensor([.5, .5, 1., 1.])).sum(1)/3.
    air = ((1-large)*small_air+large*(.25*missing[:, :2].mean(1)
                                                        +.75*rear_unled))
    rel = relative_velocity.norm(dim=-1)
    leg_motion = bounded((rel-.06).clamp_min(0)/.25)
    motion = ((leg_motion*leg_motion.new_tensor([.5, .5, 1., 1.])).sum(1)/3.)*(1-.75*large)
    extension = bounded((reach.abs()-.12).clamp_min(0)/.10).mean(1)
    total = front_moment+rear_moment
    front_share = front_moment/total.clamp_min(1.)
    # Strongest while accelerating into a turn. Keep a weaker preference
    # at steady yaw; passive rear rolling does not require positive moment.
    rear_drive = bounded((.60-front_share).clamp_min(0)/.30)
    rear_drive *= (total > 2.)*torch.where(progress < .8, 1., .3)
    cost = (1.1*drift*(.6+.4*pure)+.9*backward+.8*air
            +.35*motion+.25*extension+.35*pose+.25*rear_drive)
    quality = progress*tracking*(.65+.35*(1-pose))
    return dict(cost=-cost*active, quality=quality*active, large=large,
        backward=backward, drift=drift, pose=pose, air=air,
        yaw_error=(yaw_rate-yaw).abs(), progress=progress, front_share=front_share,
        rear_unled=rear_unled)


def touchdown_risk(force_peak, previous_force, vertical_speed, previous_speed, touchdown):
    """Per-touchdown loading proxy, including pre-contact downward speed."""
    speed = torch.maximum(-vertical_speed, -previous_speed)
    descent = bounded((speed-.10).clamp_min(0)/.35)
    loading = bounded((force_peak-previous_force-120.).clamp_min(0)/250.)
    risk = (.6*descent+.4*loading)*touchdown
    return (risk*risk.new_tensor([.5, .5, 1., 1.])).sum(1)/3.
