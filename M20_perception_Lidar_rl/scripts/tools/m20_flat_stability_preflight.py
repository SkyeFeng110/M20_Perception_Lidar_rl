"""Identity checks for fresh 50498 initialization or same-stage continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def validate(checkpoint, replay, baseline, resume=False):
    meta = json.loads(replay.with_suffix('.json').read_text())
    identity = {k: meta[k] for k in ('protocol', 'source_sha256', 'replay_sha256')}
    if meta['protocol'] != 'flat_stability_50498_v1' or meta['replay_sha256'] != hashlib.sha256(replay.read_bytes()).hexdigest():
        raise ValueError('Invalid stair protection replay')
    if {s['scene'] for s in meta['sources']} != {'ascent', 'descent'}:
        raise ValueError('Missing ascent/descent protection')
    with np.load(replay, allow_pickle=False) as d:
        assert d['observation'].shape == (meta['samples'], 63)
        assert d['action'].shape == (meta['samples'], 16)
        assert np.isfinite(d['observation']).all() and np.isfinite(d['action']).all()
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('m20_flat_stability') != identity:
            raise ValueError('Use fresh initialization for 50498; resume requires this new stage')
    elif hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta['source_sha256']:
        raise ValueError('Fresh stage must initialize from the measured stair teacher')
    for name in ('ascent', 'baseline_long'):
        manifest = json.loads((baseline/name/'manifest.json').read_text())
        if manifest['checkpoint_sha256'] != meta['source_sha256']:
            raise ValueError('Reference checkpoint mismatch')
        json.loads((baseline/name/'results.json').read_text())
    rows = json.loads((baseline/'descent/results.json').read_text())
    if len(rows) != 4 or not all(r['goal_reached'] and not r['fell'] and not r['observed_bypass'] for r in rows):
        raise ValueError('Invalid descent baseline')
    for r in rows:
        m = baseline/'descent'/f"descent_{r['tread']}_{r['command_speed']}"/'manifest.json'
        if json.loads(m.read_text())['checkpoint_sha256'] != meta['source_sha256']:
            raise ValueError('Descent baseline identity mismatch')
    return dict(status='flat_stability_preflight_passed', replay_samples=meta['samples'],
                stair_cases=len(meta['sources']), optimizer='resume' if resume else 'fresh',
                no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'replay', 'baseline'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    print(json.dumps(validate(**vars(p.parse_args())), indent=2))
