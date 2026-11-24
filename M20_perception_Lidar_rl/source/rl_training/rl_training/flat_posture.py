"""Moving flat stance and finite command-onset costs; no inference changes."""
import torch


def profiles(indices, elapsed):
    profile = indices.remainder(8)
    focused = profile < 6
    nominal = focused & (profile.remainder(2) == 0)
    partial = (profile == 2) | (profile == 3)
    light = (profile == 4) | (profile == 5)
    speed = torch.where(light | (partial & (elapsed < 2.5)), .3, 1.)
    return focused, nominal, speed*(elapsed >= 1.)


def bounded(value):
    return value.square()/(1+value.square())


def posture_terms(reach, yaw, lateral, speed, command, pitch, pitch_rate, wheel_y, active):
    moving = active & (command > .1)
    excess = (reach.abs()-.10).clamp_min(0)/.08
    per_leg = bounded(excess)
    shape = .5*per_leg.mean(1)+.5*per_leg.amax(1)
    tilt = bounded((pitch.abs()-.03491).clamp_min(0)/.05236)
    rate = bounded((pitch_rate.abs()-.35).clamp_min(0)/.60)
    outward = wheel_y*wheel_y.new_tensor([1., -1., 1., -1.])
    narrow = bounded((.20-outward).clamp_min(0)/.06).amax(1)
    attitude = .4*tilt+.35*rate+.25*narrow
    route = .5*bounded(yaw/.10)+.5*bounded(lateral/.15)
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    # Moving with a compact stance earns credit. Standing earns none; native
    # velocity/progress/deficit rewards remain active at the requested speed.
    cost = (1.2*shape+.6*attitude+.2*route)*moving
    quality = progress*(.6*(1-shape)+.25*(1-attitude)+.15*(1-route))*moving
    return dict(cost=cost, quality=quality, shape=shape, attitude=attitude,
                reach=reach.abs().amax(1), pitch=pitch.abs(), pitch_rate=pitch_rate.abs(),
                speed=speed, yaw=yaw.abs(), lateral=lateral.abs(), active=moving)


def onset_step(window, peaks, previous_command, command, action, previous_action,
               pitch_rate, wheel_y, active, reset, dt):
    """Independent peak budgets per acceleration event, not a per-frame tax.

    A single dangerous 20ms impulse must not disappear inside a 20s episode.
    Costs are charged only for new peaks during 0.6s after a command increase;
    no further charge for holding a peak or harmless repeated actions.
    """
    trigger = active & (command > .1) & (command-previous_command >= .15) & ~reset
    window = torch.where(reset, 0., (window-dt).clamp_min(0))
    window = torch.where(trigger, .6, window)
    peaks = torch.where((reset | trigger)[:, None], 0., peaks)
    delta = action-previous_action
    # Physical units: sagittal radians, wheel rad/s. Ordinary exploration and
    # small useful corrections fit inside the free interval.
    leg = (delta[:, [1, 2, 4, 5, 7, 8, 10, 11]].abs()*.25).amax(1)
    drive = (delta[:, 12:].abs()*5.).amax(1)
    outward = wheel_y*wheel_y.new_tensor([1., -1., 1., -1.])
    risk = torch.stack((bounded((leg-.14).clamp_min(0)/.21),
                        bounded((drive-2.).clamp_min(0)/5.),
                        bounded((pitch_rate.abs()-.35).clamp_min(0)/.60),
                        bounded((.20-outward).clamp_min(0)/.06).amax(1)), 1)
    enabled = active & (window > 0) & ~reset
    increase = (risk-peaks).clamp_min(0)*enabled[:, None]
    peaks = torch.where(enabled[:, None], torch.maximum(peaks, risk), peaks)
    weights = action.new_tensor([.6, .4, .6, .4])
    return window, peaks, (increase*weights).sum(1)/dt
