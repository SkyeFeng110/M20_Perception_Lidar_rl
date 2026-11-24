"""Training-only preservation of successful sequential support, native 63D ABI."""
import torch


def moving_common_drive_loss(predicted, target, command, tolerance=.05):
    """Retain mean drive only. A biased teacher cannot constrain steering."""
    error = ((predicted[:, 12:]-target[:, 12:].detach()).mean(1).abs()-tolerance).clamp_min(0).square()
    moving = command.abs() > .1
    return (error*moving).sum()/moving.sum().clamp_min(1)


def successful_support_loss(predicted, target, tolerance=.05):
    """Retain moving sagittal leg timing and common drive; release steering.

    The target is a successful moving teacher, never zero action. Hipx and
    wheel differentials remain free so retaining support does not lock in the
    teacher's yaw defect. This is a soft constraint, with no frozen student.
    """
    delta = predicted-target.detach()
    indices = [1, 2, 4, 5, 7, 8, 10, 11]
    error = (delta[:, indices].abs()-tolerance).clamp_min(0).square()
    weights = delta.new_tensor([1., 1., 1., 1., .5, .5, .5, .5])
    common = (delta[:, 12:].mean(1).abs()-tolerance).clamp_min(0).square()
    return (error.mul(weights).sum(1)/6.+.5*common).mean()


def local_mean_drift(predicted, target, std):
    """Gaussian mean KL at fixed rollout variance, independent of learned std."""
    return .5*((predicted-target.detach())/std.detach().clamp_min(.1)).square().sum(1).mean()


def forward_route_quality(speed, command, yaw, lateral, yaw_rate=None, lateral_speed=None):
    """Positive progress AND straightness credit: standing still earns zero."""
    progress = (speed/command.clamp_min(.1)).clamp(0, 1)
    quality = .5/(1+(yaw/.20).square())+.5/(1+(lateral/.30).square())
    if yaw_rate is not None and lateral_speed is not None:
        rates = .5/(1+(yaw_rate/.15).square())+.5/(1+(lateral_speed/.15).square())
        quality = .5*(quality+rates)
    return progress*quality*(command > .1)
