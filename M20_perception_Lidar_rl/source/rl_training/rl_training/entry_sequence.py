"""Bounded geometry objectives on verified flat terrain only."""
import torch


def flat_motion_terms(reach, yaw, lateral, speed, command, pitch, pitch_rate, wheel_y, active):
    moving = active & (command > .1)
    extension = ((reach.abs()-.10).clamp_min(0)/.15).square().mean(1)
    shape = extension/(1+extension)
    route = .5*(yaw/.10).square()/(1+(yaw/.10).square())
    route += .5*(lateral/.15).square()/(1+(lateral/.15).square())
    tilt = ((pitch.abs()-.05236).clamp_min(0)/.08727).square()
    rate = ((pitch_rate.abs()-.35).clamp_min(0)/.75).square()
    outward = wheel_y*wheel_y.new_tensor([1., -1., 1., -1.])
    narrow = ((.18-outward).clamp_min(0)/.08).square().mean(1)
    attitude = .4*tilt/(1+tilt)+.3*rate/(1+rate)+.3*narrow/(1+narrow)
    # The cost cannot dominate native locomotion; every positive quality term
    # requires actual path progress. No reduction of the requested speed.
    cost = (.18*shape+.18*route+.14*attitude)*moving
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    quality = progress*(.35*(1-shape)+.35*(1-route)+.30*(1-attitude))*moving
    return dict(cost=cost, quality=quality, reach=reach.abs().amax(1), yaw=yaw.abs(),
                lateral=lateral.abs(), speed=speed, active=moving)
