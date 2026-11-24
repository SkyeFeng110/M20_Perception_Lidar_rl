"""B1 is free to relearn; retain only measured later gait and flat behavior."""
import json
from pathlib import Path
import numpy as np
import torch
from .v111_step_retention_runner import StepRetentionPPO
from .step_retention import moving_common_drive_loss

PROTOCOL = 'measured_four_wheel_b2_plus_120ms_v1'


class EntrySequencePPO(StepRetentionPPO):
    def __init__(self, *args, entry_reference_results, flat_reference_results,
                 flat_reference_checkpoint, forward_reference_results, **kwargs):
        replay_path = Path(kwargs['replay_path'])
        metadata = json.loads(replay_path.with_suffix('.json').read_text())
        if metadata.get('entry_mask_protocol') != PROTOCOL:
            raise ValueError('EntrySequence requires measured four-wheel post-B2 replay')
        with np.load(replay_path, allow_pickle=False) as data:
            self._post_b2_mask = data['post_entry'].copy()
            if self._post_b2_mask.dtype != np.bool_ or self._post_b2_mask.shape != (len(data['observation']),):
                raise ValueError('Invalid post-B2 replay mask')
        super().__init__(*args, **kwargs)
        self.entry_reference_results = str(Path(entry_reference_results).resolve())
        self.flat_reference_results = str(Path(flat_reference_results).resolve())
        self.flat_reference_checkpoint = str(Path(flat_reference_checkpoint).resolve())
        self.forward_reference_results = str(Path(forward_reference_results).resolve())
        self.behavior_teacher_checkpoint = str(Path(kwargs['teacher_checkpoint']).resolve())
        self.post_entry_samples = sum(len(members) for members in self.success_members)
        self.base_learning_rate = float(self.learning_rate)

    def successful_reference_case(self, row):
        # The baseline skips B1. A completed flight is useful evidence only
        # after every wheel has established >=B2 support (mask below).
        return (row['terrain'] == 'ascent' and row['whole_body_top_supported']
                and row['termination'] == 'flight_complete' and not row['fell']
                and not row['stall_detected'] and not row['observed_bypass'])

    def successful_reference_mask(self, case):
        mask = torch.as_tensor(self._post_b2_mask, device=self.device)
        return super().successful_reference_mask(case) & mask

    def reference_loss(self, predicted, target, x, phase, scene):
        # No B1 leg, drive or steering targets, for either front or rear.
        # Release the seed's flat extension and steering bias. Preserve only
        # moving common drive; B2+ sagittal gait uses the separate success loss.
        flat = scene == 0
        if flat.any():
            return self.anchor_weight*moving_common_drive_loss(
                predicted[flat], target[flat], x[flat, 6], tolerance=.1)
        return predicted.sum()*0.

    def behavior_feedback(self, regressions):
        # Missing B1 targets and timing changes must not ratchet the global
        # learning rate down. Stronger retention is reserved for lost physical
        # capability; clean evaluations restore both controls toward baseline.
        critical = [r for r in regressions if any(key in r for key in (
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
        return dict(learning_rate=self.learning_rate,
                    success_anchor_multiplier=self.behavior_anchor_multiplier,
                    critical_regressions=critical, action='record_and_continue')

    def save(self):
        result = super().save()
        result['m20_entry_sequence'] = dict(protocol=PROTOCOL)
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('optimizer'):
            if loaded_dict.get('m20_entry_sequence', {}).get('protocol') != PROTOCOL:
                raise ValueError('Continuation requires an EntrySequence checkpoint; use initial-checkpoint for a fresh stage')
        return super().load(loaded_dict, load_cfg, strict)
