"""Local B1-B3 repair, with the measured 58098 flat/ascent/descent retained."""
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .early_traction import PROTOCOL, sha, action_band_loss
from .v111_route_hold_runner import RouteHoldPPO
from .v111_targeted_v2_runner import TargetedPPOV2
from .step_retention import local_mean_drift
from .targeted_reference import mirror_observation, straight_mask
from .route_hold import masked_steering_equivariance


class EarlyTractionPPO(RouteHoldPPO):
    def __init__(self, *args, protection_replay, descent_reference_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.descent_reference_results = str(Path(descent_reference_results).resolve())
        self.protection_sha256 = sha(protection_replay)
        self.descent_reference_sha256 = sha(descent_reference_results)
        meta = json.loads(Path(protection_replay).with_suffix('.json').read_text())
        if (meta.get('protocol') != PROTOCOL or meta.get('sha256') != self.protection_sha256
                or meta.get('teacher_sha256') != self.reference_sha256
                or meta.get('descent_results_sha256') != self.descent_reference_sha256):
            raise ValueError('Early handoff repair needs measured protection from its own teacher')
        for source in meta['sources']:
            for key in ('trajectory', 'manifest'):
                if sha(source[key]) != source[key+'_sha256']:
                    raise ValueError('Early handoff protection source changed: '+key)
        with np.load(protection_replay, allow_pickle=False) as d:
            obs, action, early, case = [d[k].copy() for k in ('observation', 'action', 'early_wheel', 'case')]
        n = len(obs)
        if (n < 100 or obs.shape != (n, 63) or action.shape != (n, 16)
                or early.shape != (n, 4) or early.dtype != np.bool_
                or case.shape != (n,) or not np.isfinite(obs).all() or not np.isfinite(action).all()):
            raise ValueError('Invalid handoff protection replay')
        self.protected_obs = torch.as_tensor(obs, device=self.device)
        self.protected_action = torch.as_tensor(action, device=self.device)
        self.protected_early = torch.as_tensor(early, device=self.device)
        members = [torch.as_tensor(np.flatnonzero(case == k), device=self.device) for k in np.unique(case)]
        sizes = [len(m) for m in members]
        self.protected_sizes = torch.tensor(sizes, device=self.device)
        self.protected_members = torch.stack([torch.cat((m, m[:1].expand(max(sizes)-len(m)))) for m in members])
        self.protection_losses = []

    def reference_loss(self, predicted, target, x, phase, scene):
        # Replace the inherited all-stair/common-wheel imitation. Keeping it
        # would pull early wheels back toward the very slip being repaired.
        return predicted.sum()*0.

    def extra_anchor_loss(self):
        # Deliberately replace parent replay losses, rather than layering a
        # permissive new loss over older, conflicting wheel/leg imitation.
        groups = torch.randint(len(self.protected_sizes), (self.anchor_batch_size,), device=self.device)
        offsets = (torch.rand(len(groups), device=self.device)*self.protected_sizes[groups]).long()
        ids = self.protected_members[groups, offsets]
        obs = TensorDict({'policy': self.protected_obs[ids]}, batch_size=[len(ids)])
        predicted = self.actor(obs)
        preserve = max(self.behavior_anchor_multiplier, self.route_flat_multiplier)*action_band_loss(
            predicted, self.protected_action[ids], self.protected_early[ids])
        reflected = self.actor(TensorDict({'policy': mirror_observation(obs['policy'])}, batch_size=[len(ids)]))
        steering = self.route_steering_weight*masked_steering_equivariance(
            predicted, reflected, straight_mask(obs['policy']))
        local = preserve.new_zeros(())
        if self._local_obs is not None:
            local = self.local_anchor_weight*local_mean_drift(
                self.actor(self._local_obs), self._local_target, self._local_std)
        self.protection_losses.append(preserve.detach())
        self.local_losses.append(local.detach())
        self.route_steering_losses.append(steering.detach())
        return preserve+local+steering+TargetedPPOV2.extra_anchor_loss(self)

    def update(self):
        self.protection_losses.clear()
        result = super().update()
        result['protected_58098_actions'] = float(torch.stack(self.protection_losses).mean()) if self.protection_losses else 0.
        return result

    def behavior_feedback(self, regressions):
        result = super().behavior_feedback(regressions)
        # Descent is an actual selection gate in this phase; pose/speed loss
        # there must also strengthen the protection, not merely be logged.
        if any(r.startswith('descent/') for r in regressions):
            self.behavior_anchor_multiplier = min(1.5, self.behavior_anchor_multiplier*1.1)
            result['success_anchor_multiplier'] = self.behavior_anchor_multiplier
        return result

    def _contract(self):
        return dict(protocol=PROTOCOL, protection_sha256=self.protection_sha256,
                    descent_reference_sha256=self.descent_reference_sha256,
                    teacher_sha256=self.reference_sha256)

    def save(self):
        result = super().save()
        result['m20_early_handoff_58098'] = self._contract()
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if (load_cfg is None or load_cfg.get('optimizer')) and loaded_dict.get('m20_early_handoff_58098') != self._contract():
            raise ValueError('Changed handoff objective/protection requires initial-checkpoint and fresh optimizer')
        return super().load(loaded_dict, load_cfg, strict)
