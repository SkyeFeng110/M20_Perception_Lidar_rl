"""Reward-only route retention and steering equivariance; native M20 actor ABI."""
import torch

PROTOCOL = 'm20_route_hold_56898_v1'


def latch_reference(heading, origin, valid, current_heading, position, straight, reset):
    """Keep one reference across repeated straight/idle command resamples."""
    capture = straight & (reset | ~valid)
    heading = torch.where(capture, current_heading, heading)
    origin = torch.where(capture[:, None], position, origin)
    valid = straight & (valid | capture)
    return heading, origin, valid


def route_terms(yaw, lateral, route_lateral_speed, yaw_rate, speed, command, active, ascent):
    moving = active & (command > .1)
    bounded = lambda x: x.square()/(1+x.square())
    yaw_error = bounded((yaw.abs()-.01745).clamp_min(0)/.05236)
    lateral_error = bounded((lateral.abs()-.025).clamp_min(0)/.10)
    # Corrections toward the original heading/line need room to act.
    rate_error = bounded((yaw_rate.abs()-.035).clamp_min(0)/.12)
    rate_error *= torch.where(yaw*yaw_rate < 0, .1, 1.)
    side_error = bounded((route_lateral_speed.abs()-.015).clamp_min(0)/.075)
    side_error *= torch.where(lateral*route_lateral_speed < 0, .1, 1.)
    error = .4*yaw_error+.3*lateral_error+.2*side_error+.1*rate_error
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    quality = (.45/(1+(yaw/.05236).square())
               +.3/(1+(lateral/.10).square())
               +.25/(1+(route_lateral_speed/.075).square()))
    # Commanded motion enables cost even at rest. Stopping cannot disable
    # route cost or collect the progress-dependent bonus.
    deficit = (.8-speed/command.clamp_min(.1)).clamp(0, 1)
    weight = torch.where(ascent, 1., .5)
    return dict(cost=-(weight*error+.35*deficit)*moving,
                quality=progress*quality*moving, active=moving,
                yaw=yaw.abs(), lateral=lateral.abs(),
                lateral_speed=route_lateral_speed.abs(), yaw_rate=yaw_rate.abs(), speed=speed)


def steering_equivariance(direct, reflected):
    from .targeted_reference import mirror_action
    reflected = mirror_action(reflected)
    hips = [0, 3, 6, 9]
    hip_error = (direct[:, hips]-reflected[:, hips]).square().mean()
    wheel = direct[:, 12:]-direct[:, 12:].mean(1, keepdim=True)
    mirrored_wheel = reflected[:, 12:]-reflected[:, 12:].mean(1, keepdim=True)
    return .5*hip_error+.5*(wheel-mirrored_wheel).square().mean()
