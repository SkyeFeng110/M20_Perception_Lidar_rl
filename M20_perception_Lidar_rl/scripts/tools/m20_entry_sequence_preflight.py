"""Verify the selected baseline, strict entry evidence and post-B2 replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
from behavior_selection import BehaviorSelector
from entry_selection import extra_results, diagnostics
from flat_start_selection import validated_results, validated_forward


def evidence_path(value):
    """Resolve original evidence paths against a relocated repository copy."""
    path = Path(value)
    if not path.is_absolute():
        return ROOT / path
    marker = '/M20_perception_Lidar_rl/'
    if marker in str(path):
        relocated = ROOT / str(path).split(marker, 1)[1]
        if relocated.is_file():
            return relocated
    return path


def validate(teacher, replay, native, entry, flat, production, resume=None):
    digest = hashlib.sha256(Path(teacher).read_bytes()).hexdigest()
    rows = BehaviorSelector._validated_results(teacher, native)
    extra = extra_results(entry, teacher)
    validated_results(flat, teacher)
    forward = validated_forward(production, teacher)
    if any(r['entry_transfer']['protocol'] != 'entry_transfer_v3'
           for r in diagnostics(rows+extra+forward)):
        raise ValueError('Strict continuous-handoff diagnostics must be recollected')
    metadata = json.loads(Path(replay).with_suffix('.json').read_text())
    if (metadata['source_sha256'] != digest
            or metadata['replay_sha256'] != hashlib.sha256(Path(replay).read_bytes()).hexdigest()
            or metadata.get('entry_mask_protocol') != 'measured_four_wheel_b2_plus_120ms_v1'):
        raise ValueError('Post-B2 replay identity/protocol mismatch')
    successes, offset = 0, 0
    by_key = {(r['profile'], r['terrain'], r['tread'], r['command_speed']): r for r in rows}
    with np.load(replay, allow_pickle=False) as data:
        mask = data['post_entry']
        if mask.dtype != np.bool_ or mask.shape != (metadata['samples'],):
            raise ValueError('Invalid post-B2 mask')
        if not np.isfinite(data['observation']).all() or data['observation'].shape != (len(mask), 63):
            raise ValueError('Replay ABI mismatch')
        for case in metadata['cases']:
            trajectory = evidence_path(case['trajectory'])
            if hashlib.sha256(trajectory.read_bytes()).hexdigest() != case['trajectory_sha256']:
                raise ValueError('Replay trajectory changed')
            source_manifest = evidence_path(case['manifest'])
            if (source_manifest.parent.parent != Path(native).resolve().parent
                    or hashlib.sha256(source_manifest.read_bytes()).hexdigest() != case['manifest_sha256']):
                raise ValueError('Replay must use the measured seed native suite')
            with np.load(trajectory, allow_pickle=False) as d:
                keep = d['policy_active'] & np.isin(d['phase'], [1, 2, 3])
                expected = np.zeros(len(d['time']), dtype=bool)
                if case['terrain'] == 'ascent':
                    ready = [next((i+2 for i in range(len(expected)-2)
                        if (d['supported_tread'][i:i+3, side] >= 2).all()), None)
                        for side in range(4)]
                    if all(i is not None for i in ready):
                        expected = d['time'] >= d['time'][max(ready)]+.12
                segment = mask[offset:offset+int(keep.sum())]
                if not np.array_equal(segment, expected[keep]):
                    raise ValueError('Replay includes an unfinished B1 transfer')
                row = by_key[(case['profile'], case['terrain'], case['tread'], case['command_speed'])]
                successes += bool(case['terrain'] == 'ascent' and segment.any()
                    and row['whole_body_top_supported'] and row['termination'] == 'flight_complete'
                    and not row['fell'] and not row['stall_detected'] and not row['observed_bypass'])
                offset += int(keep.sum())
    if not successes or offset != metadata['samples']:
        raise ValueError('No valid later-gait protection samples')
    if resume:
        import torch
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('m20_entry_sequence', {}).get('protocol') != metadata['entry_mask_protocol']:
            raise ValueError('Resume must come from this EntrySequence stage')
        state = checkpoint.get('m20_step_retention', {})
        if state.get('reference_sha256') != digest or state.get('replay_sha256') != metadata['replay_sha256']:
            raise ValueError('Resume protection assets do not match')
    return dict(status='entry_sequence_preflight_passed', teacher=str(Path(teacher).resolve()),
        sha256=digest, replay_samples=metadata['samples'], post_b2_samples=int(mask.sum()),
        completed_ascent_reference_cases=successes, no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'replay', 'native', 'entry', 'flat', 'production'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    print(json.dumps(validate(**vars(args)), indent=2))
