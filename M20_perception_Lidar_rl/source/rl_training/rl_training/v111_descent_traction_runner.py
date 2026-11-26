"""V2: independent terrain retention and descent/low-mu feedback."""
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .early_traction import sha
from .descent_traction import PROTOCOL, SAGITTAL, retention_loss, feedback_groups
from .front_turn import PROTOCOL as TURN_PROTOCOL
from .v111_early_traction_runner import EarlyTractionPPO
from .v111_targeted_v2_runner import TargetedPPOV2
from .step_retention import local_mean_drift
from .targeted_reference import mirror_observation, mirror_action, straight_mask, masked_equivariance_loss
from .route_hold import masked_steering_equivariance


class DescentTractionPPO(EarlyTractionPPO):
    repair_protocol = PROTOCOL

    def __init__(self, *args, repair_replay, **kwargs):
        super().__init__(*args, **kwargs)
        self.repair_replay_sha256 = sha(repair_replay)
        meta = json.loads(Path(repair_replay).with_suffix('.json').read_text())
        if (meta.get('protocol') != PROTOCOL or meta.get('sha256') != self.repair_replay_sha256
                or meta.get('teacher_sha256') != self.reference_sha256
                or meta.get('source_replay_sha256') != self.protection_sha256):
            raise ValueError('V2 preservation replay identity mismatch')
        with np.load(repair_replay, allow_pickle=False) as d:
            x, y, early, case, scope = [d[k].copy() for k in ('observation', 'action', 'early_wheel', 'case', 'scope')]
        n = len(x)
        if (x.shape != (n, 63) or y.shape != (n, 16) or early.shape != (n, 4)
                or early.dtype != np.bool_ or case.shape != (n,) or scope.shape != (n,)
                or set(scope) != {0, 1, 2} or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError('Invalid V2 replay ABI or terrain coverage')
        self.v2_obs = torch.as_tensor(x, device=self.device)
        self.v2_action = torch.as_tensor(y, device=self.device)
        self.v2_early = torch.as_tensor(early, device=self.device)
        self.v2_scope = torch.as_tensor(scope, device=self.device)
        members, group_scope = [], []
        for i in range(3):
            for key in np.unique(case[scope == i]):
                members.append(torch.as_tensor(np.flatnonzero((case == key) & (scope == i)), device=self.device))
                group_scope.append(i)
        sizes = [len(m) for m in members]
        self.v2_sizes = torch.tensor(sizes, device=self.device)
        self.v2_members = torch.stack([torch.cat((m, m[:1].expand(max(sizes)-len(m)))) for m in members])
        self.v2_groups = [torch.tensor([k for k, s in enumerate(group_scope) if s == i], device=self.device) for i in range(3)]
        self.keep_multiplier = dict(flat=1., ascent=1., descent=1.)
        self.clean_streak = dict(flat=0, ascent=0, descent=0, traction=0, route=0)
        self.descent_reward_multiplier = 1.
        self.slip_reward_multiplier = 1.
        self.turn_reward_multiplier = 1.
        self.turn_clean_streak = 0
        self.v2_losses = {}

    def _sync_traction_feedback(self):
        super()._sync_traction_feedback()
        if self._reward_env is not None and hasattr(self, 'descent_reward_multiplier'):
            self._reward_env._m20_v2_descent_multiplier = self.descent_reward_multiplier
            self._reward_env._m20_v2_slip_multiplier = self.slip_reward_multiplier
            self._reward_env._m20_v2_turn_multiplier = self.turn_reward_multiplier

    def extra_anchor_loss(self):
        loss = torch.zeros((), device=self.device)
        samples = []
        per_scope = max(16, self.anchor_batch_size//3)
        for i, name in enumerate(('flat', 'ascent', 'descent')):
            groups = self.v2_groups[i]
            chosen = groups[torch.randint(len(groups), (per_scope,), device=self.device)]
            offset = (torch.rand(per_scope, device=self.device)*self.v2_sizes[chosen]).long()
            ids = self.v2_members[chosen, offset]
            x, target, early = self.v2_obs[ids], self.v2_action[ids], self.v2_early[ids]
            reflected = torch.rand(per_scope, device=self.device) < .5
            x = torch.where(reflected[:, None], mirror_observation(x), x)
            target = torch.where(reflected[:, None], mirror_action(target), target)
            early = torch.where(reflected[:, None], early[:, [1, 0, 3, 2]], early)
            prediction = self.actor(TensorDict({'policy': x}, batch_size=[per_scope]))
            # Flat/ascent/descent are equally represented, then cases within
            # each terrain are balanced. Descent no longer gets 4/28 cases.
            keep = (2. if i == 0 else 3.)*self.keep_multiplier[name]*retention_loss(
                prediction, target, early, self.v2_scope[ids])
            self.v2_losses.setdefault('v2_'+name+'_keep', []).append(keep.detach())
            loss = loss+keep/3.
            samples.append(x)
        x = torch.cat(samples)
        obs = TensorDict({'policy': x}, batch_size=[len(x)])
        direct = self.actor(obs)
        reflected = self.actor(TensorDict({'policy': mirror_observation(x)}, batch_size=[len(x)]))
        steering = self.route_steering_weight*masked_steering_equivariance(direct, reflected, straight_mask(x))
        self.route_steering_losses.append(steering.detach())
        local = loss.new_zeros(())
        live = loss.new_zeros(())
        if self._local_obs is not None:
            prediction = self.actor(self._local_obs)
            local = self.local_anchor_weight*local_mean_drift(prediction, self._local_target, self._local_std)
            with torch.no_grad():
                target = self.teacher(self._local_obs)
            # Live states also preserve sagittal gait. HipX and wheel speeds
            # remain available for route correction and low-mu traction.
            x = self._local_obs['policy']
            turning = x[:, 8].abs() >= .08
            error = ((prediction[:, SAGITTAL]-target[:, SAGITTAL]).abs()-.06).clamp_min(0).square().mean(1)
            live = (error*~turning).sum()/(~turning).sum().clamp_min(1)
            # Do not copy the teacher's defective turn back into the policy.
            # Reflection swaps left/right commands and legs, never front/rear.
            mirrored = self.actor(TensorDict({'policy': mirror_observation(x)}, batch_size=[len(x)]))
            steering = steering+.15*masked_equivariance_loss(prediction, mirrored, turning)
        self.local_losses.append(local.detach())
        self.v2_losses.setdefault('v2_live_sagittal', []).append(live.detach())
        return loss+steering+local+live+TargetedPPOV2.extra_anchor_loss(self)

    def update(self):
        self.v2_losses.clear()
        result = super().update()
        result.update({k:float(torch.stack(v).mean()) for k, v in self.v2_losses.items()})
        return result

    def behavior_feedback(self, regressions):
        # Do not call the parent feedback: its clean decay cancelled the
        # descent gain on every eval without a fall, leaving it stuck at 1.1.
        groups = feedback_groups(regressions)
        turn_failed = any(r.startswith('turn/') for r in regressions)
        self.turn_clean_streak = 0 if turn_failed else self.turn_clean_streak+1
        if turn_failed:
            self.turn_reward_multiplier = min(1.3, self.turn_reward_multiplier*1.05)
        elif self.turn_clean_streak >= 3:
            self.turn_reward_multiplier = max(1., self.turn_reward_multiplier/1.1)
        for key, failed in groups.items():
            self.clean_streak[key] = 0 if failed else self.clean_streak[key]+1
        for key in self.keep_multiplier:
            if groups[key]:
                self.keep_multiplier[key] = min(3., self.keep_multiplier[key]*1.2)
            elif self.clean_streak[key] >= 3:
                self.keep_multiplier[key] = max(1., self.keep_multiplier[key]/1.1)
        if turn_failed or any(groups[k] for k in ('flat', 'ascent', 'descent')):
            self.learning_rate = max(.5*self.base_learning_rate, self.learning_rate*.85)
        elif all(self.clean_streak[k] >= 3 for k in ('flat', 'ascent', 'descent')):
            self.learning_rate = min(self.base_learning_rate, self.learning_rate*1.05)
        if groups['descent']:
            self.descent_reward_multiplier = min(1.5, self.descent_reward_multiplier*1.1)
        elif self.clean_streak['descent'] >= 3:
            self.descent_reward_multiplier = max(1., self.descent_reward_multiplier/1.1)
        if groups['traction']:
            self.slip_reward_multiplier = min(1.4, self.slip_reward_multiplier*1.05)
        elif self.clean_streak['traction'] >= 3:
            self.slip_reward_multiplier = max(1., self.slip_reward_multiplier/1.1)
        self.route_steering_weight = (min(.3, self.route_steering_weight*1.1) if groups['route'] else
            max(self.route_base_steering_weight, self.route_steering_weight/1.05)
            if self.clean_streak['route'] >= 3 else self.route_steering_weight)
        self.route_flat_multiplier = min(1.5, self.keep_multiplier['flat'])
        self.behavior_anchor_multiplier = min(1.5, max(self.keep_multiplier.values()))
        self.early_traction_multiplier = self.slip_reward_multiplier
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
        self._sync_traction_feedback()
        return dict(learning_rate=self.learning_rate, action='record_and_continue',
            success_anchor_multiplier=self.behavior_anchor_multiplier,
            route_flat_multiplier=self.route_flat_multiplier, route_steering_weight=self.route_steering_weight,
            early_traction_multiplier=self.early_traction_multiplier,
            v2_keep_multipliers=dict(self.keep_multiplier), v2_failure_groups=groups,
            v2_descent_reward_multiplier=self.descent_reward_multiplier,
            v2_slip_reward_multiplier=self.slip_reward_multiplier, v2_turn_reward_multiplier=self.turn_reward_multiplier)

    def _v2_contract(self):
        return dict(protocol=PROTOCOL, repair_replay_sha256=self.repair_replay_sha256,
                    teacher_sha256=self.reference_sha256, turn_protocol=TURN_PROTOCOL)

    def save(self):
        result = super().save()
        result['m20_descent_traction_v2'] = dict(contract=self._v2_contract(),
            keep_multiplier=self.keep_multiplier, clean_streak=self.clean_streak,
            descent_reward_multiplier=self.descent_reward_multiplier,
            slip_reward_multiplier=self.slip_reward_multiplier,
            turn_reward_multiplier=self.turn_reward_multiplier, turn_clean_streak=self.turn_clean_streak)
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        resume = load_cfg is None or load_cfg.get('optimizer')
        state = loaded_dict.get('m20_descent_traction_v2', {})
        if resume and state.get('contract') != self._v2_contract():
            raise ValueError('V2 descent/traction objective requires an initial checkpoint and fresh optimizer')
        result = super().load(loaded_dict, load_cfg, strict)
        if resume:
            self.keep_multiplier = {k:float(v) for k, v in state['keep_multiplier'].items()}
            self.clean_streak = {k:int(v) for k, v in state['clean_streak'].items()}
            self.descent_reward_multiplier = float(state['descent_reward_multiplier'])
            self.slip_reward_multiplier = float(state['slip_reward_multiplier'])
            self.turn_reward_multiplier = float(state['turn_reward_multiplier'])
            self.turn_clean_streak = int(state['turn_clean_streak'])
            if (set(self.keep_multiplier) != {'flat', 'ascent', 'descent'}
                    or any(not 1. <= v <= 3. for v in self.keep_multiplier.values())
                    or not 1. <= self.descent_reward_multiplier <= 1.5
                    or not 1. <= self.slip_reward_multiplier <= 1.4
                    or not 1. <= self.turn_reward_multiplier <= 1.3):
                raise ValueError('Invalid V2 feedback state')
        self._sync_traction_feedback()
        return result
