"""Targeted V2 runner: unfreeze recovery capacity with bounded exploration."""
import math
import torch

from .v111_targeted_runner import TargetedPPO


class TargetedPPOV2(TargetedPPO):
    """Allow all actor layers to relearn while retaining explicit safeguards.

    The teacher anchor remains active, native velocity/progress rewards remain
    active in the environment, and a lower bound prevents exploration variance
    from collapsing into a less-moving policy. No action magnitude penalty is
    added and the actor ABI is unchanged.
    """

    def __init__(self, *args, std_floor_leg=.18, std_floor_wheel=.35, **kwargs):
        if (not math.isfinite(std_floor_leg) or not 0 < std_floor_leg <= .30
            or not math.isfinite(std_floor_wheel) or not 0 < std_floor_wheel <= .60):
            raise ValueError('Invalid targeted exploration floors')
        super().__init__(*args, **kwargs)
        if (getattr(self.actor.distribution, 'std_type', None) != 'log'
            or self.actor.distribution.log_std_param.shape != (16,)):
            raise ValueError('Targeted V2 requires the native 16D log-standard-deviation actor')
        for parameter in self.actor.parameters():
            parameter.requires_grad_(True)
        self.std_floor_leg = float(std_floor_leg)
        self.std_floor_wheel = float(std_floor_wheel)
        self._std_floor_hook = self.optimizer.register_step_post_hook(self._project_std_floor)

    def _log_std_floor(self):
        return self.actor.distribution.log_std_param.new_tensor(
            [self.std_floor_leg]*12+[self.std_floor_wheel]*4).log()

    def _project_std_floor(self, optimizer=None, args=None, kwargs=None):
        with torch.no_grad():
            self.actor.distribution.log_std_param.clamp_(min=self._log_std_floor())

    def act(self, *args, **kwargs):
        # Also applies after loading a checkpoint with smaller exploration.
        # Deployment still uses the unchanged deterministic actor mean.
        self._project_std_floor()
        return super().act(*args, **kwargs)

    def extra_anchor_loss(self):
        log_std = self.actor.distribution.log_std_param
        floor = self._log_std_floor()
        # A soft barrier supplements the post-step floor; it never
        # penalizes a moving mean action or asks the robot to hold still.
        return .02 * torch.relu(floor-log_std).square().mean()

    def update(self):
        losses = super().update()
        std = self.actor.distribution.log_std_param.detach().exp()
        losses['exploration_leg_std_min'] = std[:12].min().detach()
        losses['exploration_wheel_std_min'] = std[12:].min().detach()
        return losses
