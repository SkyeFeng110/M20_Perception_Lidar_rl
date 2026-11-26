"""Flat correction with full-action retention of the user's 50498 stair gait."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .v111_entry_sequence_runner import EntrySequencePPO
from .flat_stability import PROTOCOL, stair_action_loss


class FlatStabilityPPO(EntrySequencePPO):
    def __init__(self, *args, stair_replay_path, stair_anchor_weight=2., leg_target_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(stair_replay_path)
        meta = json.loads(path.with_suffix('.json').read_text())
        if (meta.get('protocol') != PROTOCOL or meta.get('replay_sha256') !=
                hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Flat stability stair replay identity mismatch')
        if {s['scene'] for s in meta['sources']} != {'ascent', 'descent'}:
            raise ValueError('Both ascent and descent protection are required')
        with np.load(path, allow_pickle=False) as d:
            x, y, groups = d['observation'], d['action'], d['case_id']
        if (x.shape != (meta['samples'], 63) or y.shape != (len(x), 16)
                or groups.shape != (len(x),) or not np.isfinite(x).all()
                or not np.isfinite(y).all()):
            raise ValueError('Invalid stair protection tensors')
        self.stair_x = torch.as_tensor(x, device=self.device)
        self.stair_y = torch.as_tensor(y, device=self.device)
        self.stair_members = [torch.as_tensor(np.flatnonzero(groups == i), device=self.device)
                              for i in range(len(meta['sources']))]
        if any(len(m) == 0 for m in self.stair_members) or not 0 < stair_anchor_weight <= 10:
            raise ValueError('Invalid stair reference cases/weight')
        self.stair_anchor_weight = float(stair_anchor_weight)
        self.stair_identity = {k: meta[k] for k in ('protocol', 'source_sha256', 'replay_sha256')}
        if leg_target_limits:
            names = [f'{leg}_{joint}_joint' for leg in ('fl', 'fr', 'hl', 'hr')
                     for joint in ('hipx', 'hipy', 'knee')]
            bounds = self.stair_y.new_tensor([leg_target_limits[name] for name in names])
            # Native nominal/scale contract is checked by the training preflight.
            nominal = self.stair_y.new_tensor([0., -.3, .6]*2+[0., .3, -.6]*2)
            scale = self.stair_y.new_tensor([.125, .25, .25]*4)
            lower, upper = (bounds[:, 0]-nominal)/scale, (bounds[:, 1]-nominal)/scale
            # Retention must not pull the new policy back outside the hard limits.
            self.stair_y = self.stair_y.clone()
            self.stair_y[:, :12] = self.stair_y[:, :12].clamp(min=lower, max=upper)
            self.stair_identity['leg_target_limits'] = {
                name: list(leg_target_limits[name]) for name in names}
        self.stair_anchor_losses = []

    def reference_loss(self, predicted, target, x, phase, scene):
        # Old flat mean-wheel imitation can resist the new acceleration and
        # stance solution. All stair means now come from 50498 below.
        return predicted.sum()*0.

    def support_reference_loss(self, predicted, target):
        # Do not simultaneously pull the same stair action toward 49998.
        return predicted.sum()*0.

    def extra_anchor_loss(self):
        local = super().extra_anchor_loss()
        choices = torch.randint(len(self.stair_members), (self.anchor_batch_size,), device=self.device)
        indices = torch.empty_like(choices)
        for case, members in enumerate(self.stair_members):
            mask = choices == case
            indices[mask] = members[torch.randint(len(members), (int(mask.sum()),), device=self.device)]
        obs = TensorDict({'policy': self.stair_x[indices]}, batch_size=[len(indices)])
        stair = self.stair_anchor_weight*self.behavior_anchor_multiplier*stair_action_loss(
            self.actor(obs), self.stair_y[indices])
        self.stair_anchor_losses.append(float(stair.detach()))
        return local+stair

    def update(self):
        self.stair_anchor_losses.clear()
        losses = super().update()
        losses['stair_50498_anchor'] = sum(self.stair_anchor_losses)/max(1, len(self.stair_anchor_losses))
        return losses

    def save(self):
        result = super().save()
        result['m20_flat_stability'] = self.stair_identity
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('optimizer'):
            if loaded_dict.get('m20_flat_stability') != self.stair_identity:
                raise ValueError('Resume requires matching FlatStability objective and stair replay')
        return super().load(loaded_dict, load_cfg, strict)
