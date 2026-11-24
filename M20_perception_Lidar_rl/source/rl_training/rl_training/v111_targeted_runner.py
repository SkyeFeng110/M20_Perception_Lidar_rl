"""Targeted mean-policy correction; no inference-time wrapper or ABI change."""
import math
from itertools import chain
import torch
from tensordict import TensorDict
from .v111_recovery_runner import RecoveryPPO
from .recovery_reference import reference_anchor_loss, mean_reversal_loss
from .targeted_reference import (mirror_observation, correction_scales, settled_idle_mask,
    straight_mask, equivariance_loss, idle_wheel_loss, flat_workspace_loss, coherent_triplets)


class TargetedPPO(RecoveryPPO):
    def __init__(self, *args, symmetry_weight=.05, idle_mean_weight=.10, flat_mean_weight=.03, **kwargs):
        if (not math.isfinite(symmetry_weight) or not 0 < symmetry_weight <= .05
            or not math.isfinite(idle_mean_weight) or not 0 < idle_mean_weight <= .10
            or not math.isfinite(flat_mean_weight) or not 0 < flat_mean_weight <= .03):
            raise ValueError('Targeted correction budget exceeded')
        super().__init__(*args, **kwargs)
        self.deployment_triplets = coherent_triplets(self.replay_case, self.replay_phase)
        if not len(self.deployment_triplets):
            raise ValueError('Targeted quietness needs contiguous deployment replay')
        self.symmetry_weight, self.idle_mean_weight = symmetry_weight, idle_mean_weight
        self.flat_mean_weight = flat_mean_weight
        self.symmetry_losses, self.idle_losses = [], []
        self.workspace_losses = []
        self.extra_losses = []

    def extra_anchor_loss(self):
        """Optional bounded V2 auxiliary loss; V1 keeps the same objective."""
        return torch.zeros((), device=self.device)

    def reference_loss(self, predicted, target, x, phase, scene):
        legs, wheels = correction_scales(x, phase)
        return reference_anchor_loss(predicted, target, x, scene,
            self.front_weight, self.anchor_weight, **self.anchor_options,
            leg_scale=legs, wheel_scale=wheels)

    def reference_batch_context(self, indices):
        """Optional physical replay labels for stage-specific reference masks."""

    def _anchor_step(self, optimizer, args, kwargs):
        choices = torch.randint(len(self.replay_group_sizes), (self.anchor_batch_size,), device=self.device)
        offsets = (torch.rand(self.anchor_batch_size, device=self.device)*self.replay_group_sizes[choices]).long()
        indices = self.replay_group_members[self.replay_group_starts[choices]+offsets]
        x, phase = self.replay[indices], self.replay_phase[indices]
        self.reference_batch_context(indices)
        obs = TensorDict({'policy': x}, batch_size=[len(x)])
        with torch.no_grad():
            target = self.teacher(obs)
        with torch.enable_grad():
            predicted = self.actor(obs)
            loss = self.reference_loss(predicted, target, x, phase, self.replay_scene[indices])
            # Compare reflected observations, never force equal leg actions on
            # an unreflected, asymmetric stair state. Explicit turns excluded.
            straight = straight_mask(x)
            reflected = self.actor(TensorDict({'policy': mirror_observation(x)}, batch_size=[len(x)]))
            symmetry = self.symmetry_weight*equivariance_loss(predicted[straight], reflected[straight]) if straight.any() else predicted.sum()*0
            idle = self.idle_mean_weight*idle_wheel_loss(predicted, settled_idle_mask(x, phase))
            workspace = self.flat_mean_weight*flat_workspace_loss(predicted, x, self.replay_scene[indices])
            quiet = self._mean_quietness()
            extra = self.extra_anchor_loss()
            (loss+quiet+symmetry+idle+workspace+extra).backward()
        torch.nn.utils.clip_grad_norm_([p for p in chain(self.actor.parameters(), self.critic.parameters())
            if p.requires_grad], self.max_grad_norm)
        self.anchor_losses.append(float(loss.detach()))
        self.quiet_losses.append(float(quiet.detach()))
        self.symmetry_losses.append(float(symmetry.detach()))
        self.idle_losses.append(float(idle.detach()))
        self.workspace_losses.append(float(workspace.detach()))
        self.extra_losses.append(float(extra.detach()))

    def _mean_quietness(self):
        rollout = super()._mean_quietness()
        ramp = self.quiet_ramp_provider()
        if not self.quiet_mean_weight or not ramp:
            return rollout
        choice = torch.randint(len(self.deployment_triplets), (self.anchor_batch_size,), device=self.device)
        end = self.deployment_triplets[choice]
        triplet = torch.stack([self.replay[end-2], self.replay[end-1], self.replay[end]])
        x = triplet.flatten(0, 1)
        means = self.actor(TensorDict({'policy': x}, batch_size=[len(x)])).reshape(3, -1, 16)
        deployment = self.quiet_mean_weight*ramp*mean_reversal_loss(means, self.replay_scene[end].bool())
        # The total budget stays .02, shared between both state distributions.
        return .5*(rollout+deployment)

    def update(self):
        self.symmetry_losses.clear()
        self.idle_losses.clear()
        self.workspace_losses.clear()
        self.extra_losses.clear()
        losses = super().update()
        losses['straight_equivariance'] = sum(self.symmetry_losses)/max(1, len(self.symmetry_losses))
        losses['settled_idle_wheels'] = sum(self.idle_losses)/max(1, len(self.idle_losses))
        losses['flat_mean_workspace'] = sum(self.workspace_losses)/max(1, len(self.workspace_losses))
        losses['targeted_extra'] = sum(self.extra_losses)/max(1, len(self.extra_losses))
        return losses
