"""Verify RouteHold source trajectories, teacher and optimizer resume contract."""
import argparse
import json
from pathlib import Path
import numpy as np
from m20_flat_posture_preflight import validate as posture_validate, sha
from m20_entry_sequence_preflight import evidence_path

PROTOCOL = 'm20_route_hold_56898_v1'
RUN_PROTOCOL = 'm20_route_hold_early_traction_v1'


def validate(route_flat, **kwargs):
    traction_results = kwargs.pop('traction_results')
    traction_teacher = kwargs.pop('traction_teacher')
    result = posture_validate(**kwargs)
    from traction_selection import validated_results as validate_traction
    traction_rows, traction_manifest = validate_traction(
        traction_results, traction_teacher)
    if not traction_rows or not traction_manifest.get('early_tread_window'):
        raise ValueError('RouteHold requires measured B1-B3 contact-point traction baseline')
    low_friction_full = [r for r in traction_rows
        if r['terrain'] == 'ascent' and r['speed'] == 1. and r['friction'] == .6]
    if (len(low_friction_full) != 2 or any(
            not isinstance(r.get('early_ascent_slip_p95'), (int, float))
            or not r.get('early_loaded_contact_samples', 0) for r in low_friction_full)):
        raise ValueError('B1-B3 traction baseline lacks loaded 1 m/s cases on both target treads')
    path = Path(route_flat)
    meta = json.loads(path.with_suffix('.json').read_text())
    if (meta.get('protocol') != PROTOCOL or meta.get('sha256') != sha(path)
            or meta.get('teacher_sha256') != sha(kwargs['teacher'])):
        raise ValueError('Route flat preservation teacher/identity mismatch')
    observations, actions = [], []
    for source in meta['sources']:
        trajectory = evidence_path(source['trajectory'])
        manifest = evidence_path(source['manifest'])
        if sha(trajectory) != source['trajectory_sha256'] or sha(manifest) != source['manifest_sha256']:
            raise ValueError('Route flat source changed')
        m = json.loads(manifest.read_text())
        if (m.get('checkpoint_sha256') != meta['teacher_sha256'] or not m.get('physics_enabled')
                or m.get('protocol') != 'flat_start_v1' or m.get('policy_updated') is not False):
            raise ValueError('Route flat source is not measured teacher motion')
        with np.load(trajectory, allow_pickle=False) as d:
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
        if (state.get('protocol') != RUN_PROTOCOL
                or state.get('flat_replay_sha256') != sha(path)
                or state.get('traction_reference_sha256') != sha(traction_results)):
            raise ValueError('New route objective requires initial-checkpoint and fresh optimizer')
    result.update(status='route_hold_preflight_passed', route_flat_samples=meta['samples'],
                  actor_dimensions=[63, 16], critic_dimensions=568,
                  traction_cases=len(traction_rows),
                  low_friction_full_speed_early_slip=[r['early_ascent_slip_p95'] for r in low_friction_full])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'teacher', 'replay', 'compact', 'native', 'entry', 'flat',
                 'production', 'route-flat', 'traction-results', 'traction-teacher'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--joint-limits', action='store_true')
    print(json.dumps(validate(**vars(p.parse_args())), indent=2))
