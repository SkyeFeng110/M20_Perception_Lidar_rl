"""Verify RouteHold source trajectories, teacher and optimizer resume contract."""
import argparse
import json
from pathlib import Path
import numpy as np
from m20_flat_posture_preflight import validate as posture_validate, sha

PROTOCOL = 'm20_route_hold_56898_v1'


def validate(route_flat, **kwargs):
    result = posture_validate(**kwargs)
    path = Path(route_flat)
    meta = json.loads(path.with_suffix('.json').read_text())
    if (meta.get('protocol') != PROTOCOL or meta.get('sha256') != sha(path)
            or meta.get('teacher_sha256') != sha(kwargs['teacher'])):
        raise ValueError('Route flat preservation teacher/identity mismatch')
    observations, actions = [], []
    for source in meta['sources']:
        if sha(source['trajectory']) != source['trajectory_sha256'] or sha(source['manifest']) != source['manifest_sha256']:
            raise ValueError('Route flat source changed')
        m = json.loads(Path(source['manifest']).read_text())
        if (m.get('checkpoint_sha256') != meta['teacher_sha256'] or not m.get('physics_enabled')
                or m.get('protocol') != 'flat_start_v1' or m.get('policy_updated') is not False):
            raise ValueError('Route flat source is not measured teacher motion')
        with np.load(source['trajectory'], allow_pickle=False) as d:
            index = np.asarray(source['source_index'])
            if not np.array_equal(index, np.flatnonzero(d['time'] >= 1.)) or len(index) != source['samples']:
                raise ValueError('Missing measured startup/movement coverage')
            observations.append(d['observation'][index]); actions.append(d['processed_action'][index])
    with np.load(path, allow_pickle=False) as d:
        if (len(d['action']) < 300 or d['observation'].shape != (len(d['action']), 63)
                or d['action'].shape[1:] != (16,) or not np.isfinite(d['observation']).all()
                or not np.isfinite(d['action']).all()
                or not np.array_equal(d['observation'], np.concatenate(observations))
                or not np.array_equal(d['action'], np.concatenate(actions))):
            raise ValueError('Route flat replay differs from executed teacher motion')
    if kwargs.get('resume'):
        import torch
        state = torch.load(kwargs['resume'], map_location='cpu', weights_only=True).get('m20_route_hold', {})
        if state.get('protocol') != PROTOCOL or state.get('flat_replay_sha256') != sha(path):
            raise ValueError('New route objective requires initial-checkpoint and fresh optimizer')
    result.update(status='route_hold_preflight_passed', route_flat_samples=meta['samples'],
                  actor_dimensions=[63, 16], critic_dimensions=568)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'teacher', 'replay', 'compact', 'native', 'entry', 'flat', 'production', 'route-flat'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--joint-limits', action='store_true')
    print(json.dumps(validate(**vars(p.parse_args())), indent=2))
