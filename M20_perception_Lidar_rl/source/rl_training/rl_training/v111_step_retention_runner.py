"""Full actor learning with successful MuJoCo replay and local mean stability."""
import json
import math
from pathlib import Path

import torch
from tensordict import TensorDict

from .step_retention import successful_support_loss, local_mean_drift, moving_common_drive_loss
from .targeted_reference import mirror_action, mirror_observation, straight_mask, masked_equivariance_loss
from .v111_targeted_v2_runner import TargetedPPOV2


class StepRetentionPPO(TargetedPPOV2):
    def __init__(self, *args, behavior_reference_results, success_anchor_weight=2.,
                 local_anchor_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        if any(not math.isfinite(v) or v <= 0 for v in (success_anchor_weight, local_anchor_weight)):
            raise ValueError('Retention weights must be positive and finite')
        self.behavior_reference_results = str(Path(behavior_reference_results).resolve())
        self.behavior_teacher_checkpoint = str(Path(kwargs['teacher_checkpoint']).resolve())
        rows = json.loads(Path(self.behavior_reference_results).read_text())
        metadata = json.loads(Path(kwargs['replay_path']).with_suffix('.json').read_text())
        for profile in ('standing', 'keyboard'):
            manifest = json.loads((Path(self.behavior_reference_results).parent/profile/'manifest.json').read_text())
            if manifest.get('checkpoint_sha256') != self.reference_sha256 or not manifest.get('physics_enabled'):
                raise ValueError('StepRetention behavior evidence must come from its teacher')
        suites = {str((Path(self.behavior_reference_results).parent/p/'manifest.json').resolve())
                  for p in ('standing', 'keyboard')}
        def local_manifest(value):
            marker = '/M20_perception_Lidar_rl/'
            if marker in value:
                relocated = Path(__file__).resolve().parents[3] / value.split(marker, 1)[1]
                if relocated.is_file():
                    return str(relocated.resolve())
            path = Path(value)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[3] / path
            return str(path.resolve())
        if any(local_manifest(case['manifest']) not in suites for case in metadata['cases']):
            raise ValueError('StepRetention replay and success labels must use the same trajectories')
        by_key = {(r['profile'], r['terrain'], r['tread'], r['command_speed']): r for r in rows}
        success_cases = []
        for case in metadata['cases']:
            row = by_key[(case['profile'], case['terrain'], case['tread'], case['command_speed'])]
            if self.successful_reference_case(row):
                success_cases.append(case['case_id'])
        groups = [self.successful_reference_mask(case) for case in success_cases]
        self.success_members = [mask.nonzero(as_tuple=True)[0] for mask in groups if mask.any()]
        if not self.success_members:
            raise ValueError('Step retention needs successful moving ascent trajectories')
        member_sizes = [len(members) for members in self.success_members]
        max_members = max(member_sizes)
        self.success_member_table = torch.stack([
            torch.cat((members, members[:1].expand(max_members-len(members))))
            for members in self.success_members
        ])
        self.success_member_sizes = torch.tensor(member_sizes, device=self.device)
        self.success_anchor_weight = float(success_anchor_weight)
        self.local_anchor_weight = float(local_anchor_weight)
        self.behavior_anchor_multiplier = 1.
        self.behavior_selection_state = None
        self._local_obs = None
        self.success_losses, self.local_losses = [], []

    def successful_reference_case(self, row):
        return (row['terrain'] == 'ascent' and row.get('all_treads_supported')
                and row['whole_body_top_supported'] and not row['fell']
                and not row['stall_detected'] and not row['observed_bypass'])

    def successful_reference_mask(self, case):
        return (self.replay_case == case) & (self.replay_phase == 2)

    def reference_loss(self, predicted, target, x, phase, scene):
        # 47900 has a right-turn defect. The generic anchor would still pull
        # hipx and wheel differentials toward it, even at reduced weights.
        # Preserve only forward drive here; successful legs are handled below.
        return self.anchor_weight*moving_common_drive_loss(predicted, target, x[:, 6])

    def support_reference_loss(self, predicted, target):
        return successful_support_loss(predicted, target)

    def extra_anchor_loss(self):
        # Preserve each successful physical case equally, regardless of duration.
        choices = torch.randint(len(self.success_members), (self.anchor_batch_size,), device=self.device)
        offsets = (torch.rand(self.anchor_batch_size, device=self.device)
                   *self.success_member_sizes[choices]).long()
        indices = self.success_member_table[choices, offsets]
        obs = TensorDict({'policy': self.replay[indices]}, batch_size=[len(indices)])
        with torch.no_grad():
            target = self.teacher(obs)
        # Mirror targets from actual successful trajectories. Do not query a
        # biased teacher on reflected states and reintroduce its handedness.
        reflected = torch.rand(len(indices), device=self.device) < .5
        x = torch.where(reflected[:, None], mirror_observation(obs['policy']), obs['policy'])
        target = torch.where(reflected[:, None], mirror_action(target), target)
        obs = TensorDict({'policy': x}, batch_size=[len(x)])
        success = self.success_anchor_weight*self.behavior_anchor_multiplier*self.support_reference_loss(
            self.actor(obs), target)
        local = success.new_zeros(())
        if self._local_obs is not None:
            local = self.local_anchor_weight*local_mean_drift(
                self.actor(self._local_obs), self._local_target, self._local_std)
            x = self._local_obs['policy']
            straight = straight_mask(x)
            direct = self.actor(self._local_obs)
            reflected = self.actor(TensorDict({'policy': mirror_observation(x)}, batch_size=[len(x)]))
            local = local+self.symmetry_weight*masked_equivariance_loss(direct, reflected, straight)
        self.success_losses.append(success.detach())
        self.local_losses.append(local.detach())
        return super().extra_anchor_loss()+success+local

    def update(self):
        self.success_losses.clear()
        self.local_losses.clear()
        # All physical terrains may enter this local trust objective. No label
        # is guessed from absent BHD, and the old mean is captured before PPO.
        x = self.storage.observations['policy'].flatten(0, 1)
        selected = torch.randint(len(x), (self.anchor_batch_size,), device=self.device)
        self._local_obs = TensorDict({'policy': x[selected].detach()}, batch_size=[len(selected)])
        with torch.no_grad():
            self._local_target = self.actor(self._local_obs).detach()
            self._local_std = self.actor.distribution.log_std_param.detach().exp().clone()
        try:
            losses = super().update()
            with torch.no_grad():
                losses['local_mean_kl_after_update'] = local_mean_drift(
                    self.actor(self._local_obs), self._local_target, self._local_std).detach()
        finally:
            self._local_obs = None
        losses['successful_support_anchor'] = torch.stack(self.success_losses).mean() if self.success_losses else torch.zeros((), device=self.device)
        losses['local_mean_anchor'] = torch.stack(self.local_losses).mean() if self.local_losses else torch.zeros((), device=self.device)
        losses['behavior_anchor_multiplier'] = self.behavior_anchor_multiplier
        return losses

    def behavior_feedback(self, regressions):
        """Valid failures continue learning, with a bounded correction of drift."""
        if regressions:
            self.behavior_anchor_multiplier = min(4., self.behavior_anchor_multiplier*1.25)
            self.learning_rate = max(2.e-6, self.learning_rate*.8)
            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rate
        return dict(learning_rate=self.learning_rate,
                    success_anchor_multiplier=self.behavior_anchor_multiplier,
                    action='continue_with_stronger_retention' if regressions else 'continue')

    def save(self):
        state = super().save()
        state['m20_step_retention'] = dict(learning_rate=self.learning_rate,
            behavior_anchor_multiplier=self.behavior_anchor_multiplier,
            reference_sha256=self.reference_sha256, replay_sha256=self.replay_sha256,
            best_candidate=self.behavior_selection_state)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get('optimizer'):
            state = loaded_dict.get('m20_step_retention')
            if state:
                if (state['reference_sha256'] != self.reference_sha256
                        or state['replay_sha256'] != self.replay_sha256):
                    raise ValueError('StepRetention continuation must use its saved teacher and replay')
                self.learning_rate = float(state['learning_rate'])
                self.behavior_anchor_multiplier = float(state['behavior_anchor_multiplier'])
                self.behavior_selection_state = state.get('best_candidate')
                for group in self.optimizer.param_groups:
                    group['lr'] = self.learning_rate
        return result
