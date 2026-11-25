"""Retain the measured current stair gait; relearn flat stance without freezing."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .v111_step_retention_runner import StepRetentionPPO
from .step_retention import moving_common_drive_loss, successful_support_loss
from .targeted_reference import mirror_action, mirror_observation
from .joint_target_limits import joint_target_contract as current_joint_target_contract

PROTOCOL = 'flat_posture_current_stairs_compact_flat_v1'


class FlatPosturePPO(StepRetentionPPO):
    def __init__(self, *args, entry_reference_results, flat_reference_results,
                 flat_reference_checkpoint, forward_reference_results,
                 compact_replay, compact_anchor_weight=.5, joint_target_contract=None, **kwargs):
        if joint_target_contract is not None and joint_target_contract != current_joint_target_contract():
            raise ValueError('FlatPosture target bounds differ from the deployment contract')
        self.joint_target_contract = joint_target_contract
        with np.load(kwargs['replay_path'], allow_pickle=False) as data:
            self._stair_keep = data['stair_keep'].copy()
        metadata = json.loads(Path(kwargs['replay_path']).with_suffix('.json').read_text())
        if metadata.get('posture_protocol') != PROTOCOL:
            raise ValueError('FlatPosture requires its measured current-gait replay')
        if self._stair_keep.dtype != np.bool_ or self._stair_keep.shape != (metadata['samples'],):
            raise ValueError('Invalid current stair mask')
        super().__init__(*args, **kwargs)
        for key, value in dict(entry_reference_results=entry_reference_results,
                               flat_reference_results=flat_reference_results,
                               flat_reference_checkpoint=flat_reference_checkpoint,
                               forward_reference_results=forward_reference_results).items():
            setattr(self, key, str(Path(value).resolve()))
        self.behavior_teacher_checkpoint = str(Path(kwargs['teacher_checkpoint']).resolve())
        self.base_learning_rate = float(self.learning_rate)
        self.compact_sha256 = hashlib.sha256(Path(compact_replay).read_bytes()).hexdigest()
        meta = json.loads(Path(compact_replay).with_suffix('.json').read_text())
        if meta.get('protocol') != PROTOCOL or meta.get('sha256') != self.compact_sha256:
            raise ValueError('Compact moving flat reference identity mismatch')
        with np.load(compact_replay, allow_pickle=False) as d:
            self.compact_obs = torch.as_tensor(d['observation'].copy(), device=self.device)
            self.compact_action = torch.as_tensor(d['action'].copy(), device=self.device)
        if (self.compact_obs.shape != (len(self.compact_action), 63)
                or self.compact_action.shape[1:] != (16,) or len(self.compact_obs) < 50
                or not torch.isfinite(self.compact_obs).all() or not torch.isfinite(self.compact_action).all()):
            raise ValueError('Invalid compact reference ABI or empty coverage')
        if not 0 < compact_anchor_weight <= 1.:
            raise ValueError('Invalid soft compact reference weight')
        self.compact_anchor_weight = compact_anchor_weight
        self.compact_losses = []
        deployed = Path(compact_replay).parent/'deployed_stairs.npz'
        self.deployed_sha256 = hashlib.sha256(deployed.read_bytes()).hexdigest()
        dm = json.loads(deployed.with_suffix('.json').read_text())
        if dm.get('sha256') != self.deployed_sha256 or dm.get('source_sha256') != self.reference_sha256:
            raise ValueError('Deployed stair retention identity mismatch')
        with np.load(deployed, allow_pickle=False) as d:
            self.deployed_obs = torch.as_tensor(d['observation'].copy(), device=self.device)
            self.deployed_action = torch.as_tensor(d['action'].copy(), device=self.device)
        if (self.deployed_obs.shape != (len(self.deployed_action), 63)
                or self.deployed_action.shape[1:] != (16,) or not len(self.deployed_obs)
                or not torch.isfinite(self.deployed_obs).all() or not torch.isfinite(self.deployed_action).all()):
            raise ValueError('Invalid deployed stair retention data')

    def successful_reference_case(self, row):
        return (row['terrain'] == 'ascent' and row['whole_body_top_supported']
                and not row['fell'] and not row['stall_detected'] and not row['observed_bypass'])

    def successful_reference_mask(self, case):
        return super().successful_reference_mask(case) & torch.as_tensor(self._stair_keep, device=self.device)

    def reference_loss(self, predicted, target, x, phase, scene):
        # Keep drive on recorded flat states; release the teacher's extended
        # flat legs. Stair legs are protected by the separate measured replay.
        flat = scene == 0
        if flat.any():
            return self.anchor_weight*moving_common_drive_loss(
                predicted[flat], target[flat], x[flat, 6], tolerance=.1)
        return predicted.sum()*0.

    def extra_anchor_loss(self):
        inherited = super().extra_anchor_loss()
        ids = torch.randint(len(self.compact_obs), (self.anchor_batch_size,), device=self.device)
        x, target = self.compact_obs[ids], self.compact_action[ids]
        reflected = torch.rand(len(ids), device=self.device) < .5
        x = torch.where(reflected[:, None], mirror_observation(x), x)
        target = torch.where(reflected[:, None], mirror_action(target), target)
        prediction = self.actor(TensorDict({'policy': x}, batch_size=[len(ids)]))
        # Sagittal posture only: never copy the old hipx/steering bias or its
        # full-stick transient. All student layers remain trainable.
        delta = prediction[:, [1, 2, 4, 5, 7, 8, 10, 11]]-target[:, [1, 2, 4, 5, 7, 8, 10, 11]]
        compact = self.compact_anchor_weight*(delta.abs()-.08).clamp_min(0).square().mean()
        self.compact_losses.append(float(compact.detach()))
        ids = torch.randint(len(self.deployed_obs), (self.anchor_batch_size,), device=self.device)
        x, target = self.deployed_obs[ids], self.deployed_action[ids]
        reflected = torch.rand(len(ids), device=self.device) < .5
        x = torch.where(reflected[:, None], mirror_observation(x), x)
        target = torch.where(reflected[:, None], mirror_action(target), target)
        prediction = self.actor(TensorDict({'policy': x}, batch_size=[len(ids)]))
        deployed = .5*self.behavior_anchor_multiplier*successful_support_loss(prediction, target)
        return inherited+compact+deployed

    def update(self):
        self.compact_losses.clear()
        result = super().update()
        result['compact_flat_anchor'] = sum(self.compact_losses)/max(1, len(self.compact_losses))
        return result

    def behavior_feedback(self, regressions):
        # Only loss of physical ability tightens retention, and clean reports
        # restore it. Ordinary posture failures must retain learning room.
        critical = [r for r in regressions if any(k in r for k in (
            'fell', 'observed_bypass', 'stall_detected', 'lost policy_entered',
            'lost whole_body_top_supported'))]
        if critical:
            self.behavior_anchor_multiplier = min(1.5, self.behavior_anchor_multiplier*1.1)
            self.learning_rate = max(.6*self.base_learning_rate, self.learning_rate*.9)
        else:
            self.behavior_anchor_multiplier = max(1., self.behavior_anchor_multiplier/1.1)
            self.learning_rate = min(self.base_learning_rate, self.learning_rate*1.1)
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
        return dict(learning_rate=self.learning_rate, success_anchor_multiplier=self.behavior_anchor_multiplier,
                    critical_regressions=critical, action='record_and_continue')

    def save(self):
        result = super().save()
        result['m20_flat_posture'] = dict(protocol=PROTOCOL, compact_sha256=self.compact_sha256,
                                        deployed_sha256=self.deployed_sha256)
        if self.joint_target_contract is not None:
            result['m20_joint_target_limits'] = self.joint_target_contract
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('optimizer'):
            if loaded_dict.get('m20_joint_target_limits') != self.joint_target_contract:
                raise ValueError('Use initial-checkpoint for changed joint target bounds')
            expected = dict(protocol=PROTOCOL, compact_sha256=self.compact_sha256,
                            deployed_sha256=self.deployed_sha256)
            if loaded_dict.get('m20_flat_posture') != expected:
                raise ValueError('Use initial-checkpoint for a new flat posture objective')
        return super().load(loaded_dict, load_cfg, strict)
