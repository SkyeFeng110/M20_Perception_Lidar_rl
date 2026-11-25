"""Verify separate initialization, current gait and compact posture evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
from behavior_selection import BehaviorSelector
from entry_selection import extra_results
from flat_start_selection import validated_results, validated_forward
from m20_entry_sequence_preflight import evidence_path

PROTOCOL = 'flat_posture_current_stairs_compact_flat_v1'


def sha(path):
    return hashlib.sha256(evidence_path(path).read_bytes()).hexdigest()


def validate(checkpoint, teacher, replay, compact, native, entry, flat, production, resume=None, joint_limits=False):
    rows = BehaviorSelector._validated_results(teacher, native)
    extra_results(entry, teacher)
    validated_results(flat, teacher)
    validated_forward(production, teacher)
    meta = json.loads(Path(replay).with_suffix('.json').read_text())
    if (meta.get('posture_protocol') != PROTOCOL or meta['source_sha256'] != sha(teacher)
            or meta['replay_sha256'] != sha(replay)):
        raise ValueError('Current gait replay identity mismatch')
    offset, count = 0, 0
    with np.load(replay, allow_pickle=False) as d:
        if d['observation'].shape != (meta['samples'], 63) or not np.isfinite(d['observation']).all():
            raise ValueError('Invalid gait replay observations')
        mask = d['stair_keep']
        if mask.dtype != np.bool_ or mask.shape != (meta['samples'],):
            raise ValueError('Invalid stair preservation mask')
        for case in meta['cases']:
            if sha(case['trajectory']) != case['trajectory_sha256'] or sha(case['manifest']) != case['manifest_sha256']:
                raise ValueError('Gait source trajectory changed')
            with np.load(evidence_path(case['trajectory']), allow_pickle=False) as a:
                keep = a['policy_active'] & np.isin(a['phase'], [1, 2, 3])
                expected = (np.maximum.accumulate(a['wheel_position'][:, :2, 0].max(1) >= meta['stair_start']-.5)
                    if case['terrain'] == 'ascent' else np.zeros(len(keep), bool))
                if not np.array_equal(mask[offset:offset+keep.sum()], expected[keep]):
                    raise ValueError('Stair replay masks unfinished flat approach or omits entry')
                count += int(expected[keep].sum())
                offset += int(keep.sum())
    cm = json.loads(Path(compact).with_suffix('.json').read_text())
    if (cm.get('protocol') != PROTOCOL or cm['sha256'] != sha(compact)
            or cm['source_sha256'] != sha(cm['source_checkpoint'])
            or cm['trajectory_sha256'] != sha(cm['trajectory'])
            or cm['manifest_sha256'] != sha(cm['manifest'])):
        raise ValueError('Compact reference identity mismatch')
    with np.load(compact, allow_pickle=False) as d, np.load(evidence_path(cm['trajectory']), allow_pickle=False) as source:
        index = d['source_index']
        if (len(index) < 50 or d['observation'].shape != (len(index), 63)
                or d['action'].shape != (len(index), 16) or not np.isfinite(d['action']).all()
                or not np.array_equal(d['observation'], source['observation'][index])
                or not np.array_equal(d['action'], source['action'][index])
                or not (source['phase'][index] == 2).all()
                or np.abs(d['reach']).max() > .12 or (d['speed'] < .8*source['command'][index, 0]).any()):
            raise ValueError('Compact replay must use actual compact moving actions')
    if not count or offset != meta['samples']:
        raise ValueError('Missing current entry and later gait protection')
    deployed = Path(compact).parent/'deployed_stairs.npz'
    dm = json.loads(deployed.with_suffix('.json').read_text())
    if dm.get('protocol') != PROTOCOL or dm.get('sha256') != sha(deployed) or dm.get('source_sha256') != sha(teacher):
        raise ValueError('Deployed stair reference mismatch')
    with np.load(deployed, allow_pickle=False) as d:
        if d['observation'].shape != (len(d['action']), 63) or d['action'].shape[1:] != (16,):
            raise ValueError('Invalid deployed stair replay ABI')
        observations, actions = [], []
        for source in dm['sources']:
            if sha(source['trajectory']) != source['trajectory_sha256'] or sha(source['manifest']) != source['manifest_sha256']:
                raise ValueError('Deployed stair trajectory changed')
            with np.load(evidence_path(source['trajectory']), allow_pickle=False) as a:
                keep = ((a['phase'] == 2) & np.maximum.accumulate(
                    a['wheel_position'][:, :2, 0].max(1) >= meta['stair_start']-.5))
                observations.append(a['observation'][keep]); actions.append(a['action'][keep])
        if (not np.array_equal(d['observation'], np.concatenate(observations))
                or not np.array_equal(d['action'], np.concatenate(actions))):
            raise ValueError('Deployed stair replay differs from measured motion')
    if resume:
        import torch
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state.get('m20_flat_posture') != dict(protocol=PROTOCOL, compact_sha256=sha(compact), deployed_sha256=sha(deployed)):
            raise ValueError('Resume requires this flat posture objective; latest old-run checkpoint is an initial checkpoint')
        protection = state.get('m20_step_retention', {})
        if protection.get('reference_sha256') != sha(teacher) or protection.get('replay_sha256') != sha(replay):
            raise ValueError('Resume gait protection changed')
        if joint_limits:
            import importlib.util
            spec = importlib.util.spec_from_file_location('m20_limit_resume_contract',
                ROOT/'source/rl_training/rl_training/joint_target_limits.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if state.get('m20_joint_target_limits') != module.joint_target_contract():
                raise ValueError('Changed joint target limits require an initial checkpoint and a fresh optimizer')
    return dict(status='flat_posture_preflight_passed', initial_checkpoint=str(Path(checkpoint).resolve()),
        stair_teacher=str(Path(teacher).resolve()), gait_samples=count,
        compact_samples=cm['samples'], completed_ascent_cases=sum(r['whole_body_top_supported'] for r in rows),
        all_actor_layers_trainable=True, no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'teacher', 'replay', 'compact', 'native', 'entry', 'flat', 'production'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--joint-limits', action='store_true')
    print(json.dumps(validate(**vars(p.parse_args())), indent=2))
