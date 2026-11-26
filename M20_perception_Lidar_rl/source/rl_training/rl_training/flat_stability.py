"""Flat-only geometry shaping; quantities are SI, independent of Isaac."""
import torch

PROTOCOL = 'flat_stability_50498_v1'
COST_CAP = 3.9


def huber_excess(value, free, scale, cap):
    x = ((value.abs()-free).clamp_min(0)/scale)
    return torch.where(x < 1., .5*x.square(), x-.5).clamp_max(cap)


def stability_terms(reach, yaw, lateral, speed, command, pitch, pitch_rate,
                    wheel_y, flat, straight):
    # Per-wheel violations cannot cancel between front/rear or left/right.
    # Keep a 5 cm quiet extension band, so a sustained 20-30 cm reach is
    # explicitly penalized instead of only reacting near the final bound.
    legs = huber_excess(reach, .05, .10, 4.)
    shape = .5*legs.mean(1)+.5*legs.amax(1)
    tilt = huber_excess(pitch, .03491, .06545, 2.)
    rate = huber_excess(pitch_rate, .25, .50, 2.)
    outward = wheel_y*wheel_y.new_tensor([1., -1., 1., -1.])
    width = huber_excess((.19-outward).clamp_min(0), 0., .08, 2.).mean(1)
    route = .5*(yaw/.10).square()/(1+(yaw/.10).square())
    route += .5*(lateral/.15).square()/(1+(lateral/.15).square())
    moving = flat & straight & (command > .1)
    cost = (.65*shape+.45*tilt+.20*rate+.15*width)*flat+.20*route*moving
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    quality = progress*torch.exp(-shape-.5*tilt-.5*rate-.5*width)*(1-route)*moving
    return dict(cost=cost, quality=quality, reach=reach.abs().amax(1),
                yaw=yaw.abs(), lateral=lateral.abs(), speed=speed, active=flat,
                shape=shape*flat, tilt=tilt*flat, pitch_rate=rate*flat)


def stair_action_loss(predicted, target, tolerance=.03):
    """Preserve all 16 actions of the selected stair policy, no pose target."""
    return (predicted-target.detach()).abs().sub(tolerance).clamp_min(0).square().mean()
