"""V2 descent route correction, slip-tail shaping and separated retention."""
import math
import torch

PROTOCOL = 'm20_descent_low_mu_58098_v2'
SAGITTAL = [1, 2, 4, 5, 7, 8, 10, 11]
HIPX = [0, 3, 6, 9]


def descent_route_terms(yaw, lateral, side_speed, yaw_rate, speed, command, active):
    moving = active & (command > .1)
    bounded = lambda x: x.square()/(1+x.square())
    heading = bounded((yaw.abs()-math.radians(1.)).clamp_min(0)/math.radians(3.))
    line = bounded((lateral.abs()-.025).clamp_min(0)/.075)
    side = bounded((side_speed.abs()-.015).clamp_min(0)/.075)
    turn = bounded((yaw_rate.abs()-.035).clamp_min(0)/.12)
    # Returning to the original route needs lateral/yaw motion. Do not tax
    # a corrective motion as though it were further outward drift.
    side *= torch.where(lateral*side_speed < 0, .1, 1.)
    turn *= torch.where(yaw*yaw_rate < 0, .1, 1.)
    error = .45*heading+.35*line+.15*side+.05*turn
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    quality = (.5/(1+(yaw/math.radians(3.)).square())
               +.35/(1+(lateral/.075).square())
               +.15/(1+(side_speed/.075).square()))
    # Native descent support-transfer speed gating stays authoritative. No
    # new minimum speed, horizontal-body target, or simultaneous-leg target.
    return dict(cost=-error*moving, quality=progress*quality*moving,
                active=moving, yaw=yaw, lateral=lateral,
                side_speed=side_speed, yaw_rate=yaw_rate, speed=speed)


def slip_tail_terms(slip, supported, window, speed, command):
    """Resolve high slip instead of saturating all .5..1.5m/s cases alike.

    Both terms use a fixed denominator. With .70/.18 weights, the .30 load
    credit (.21) exceeds maximum slip cost (.18), so unloading loses credit.
    Cost survives zero speed; flight and non-target terrain receive no cost.
    """
    selected = supported & window & (command > .1)[:, None]
    weights = slip.new_tensor([.5, .5, 1., 1.])
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    quality = 1./(1.+(slip/.25).square())
    severity = .5*(slip/.35).clamp(0, 1)+.5*(slip/1.5).clamp(0, 1)
    risk = .5*severity+.5*severity.square()
    reward = ((.30+.70*progress[:, None]*quality)*selected*weights).sum(1)/3.
    cost = (risk*selected*weights).sum(1)/3.
    return reward, cost, selected


def retention_loss(predicted, target, early_wheel, scope):
    """Flat/full gait kept; descent steering and B1-B3 wheel drive can adapt.

    scope: 0 flat, 1 ascent, 2 descent. Unlike V1, early sagittal actions
    keep the same tight band as later stairs. Descent copies sagittal pose
    and common drive, allowing HipX and wheel differences to correct yaw.
    """
    delta = predicted-target.detach()
    sagittal = (delta[:, SAGITTAL].abs()-.04).clamp_min(0).square().mean(1)
    hips = (delta[:, HIPX].abs()-.04).clamp_min(0).square().mean(1)
    wheel = (delta[:, 12:].abs()-.12).clamp_min(0).square()
    kept = (~early_wheel).to(predicted.dtype)
    wheel = (wheel*kept).sum(1)/4.
    common = (delta[:, 12:].mean(1).abs()-.10).clamp_min(0).square()
    flat = (scope == 0).to(predicted.dtype)
    ascent = (scope == 1).to(predicted.dtype)
    descent = (scope == 2).to(predicted.dtype)
    return (sagittal+.5*hips*(flat+ascent)+.35*wheel*(flat+ascent)+.35*common*descent).mean()


def feedback_groups(failures):
    """Disjoint independent feedback, not parent decay followed by child gain."""
    flat = any(('flat/' in r or '/flat/' in r or 'flat workspace' in r) for r in failures)
    descent = any(r.startswith('descent/') for r in failures)
    ascent = any(('ascent' in r or 'skipped' in r or 'wheel support' in r or 'four-leg order' in r)
                 and not r.startswith('traction/') for r in failures)
    traction = any(r.startswith('traction') for r in failures)
    route = any(any(k in r for k in ('yaw', 'lateral', 'heading', 'route drift')) for r in failures)
    return dict(flat=flat, ascent=ascent, descent=descent, traction=traction, route=route)
