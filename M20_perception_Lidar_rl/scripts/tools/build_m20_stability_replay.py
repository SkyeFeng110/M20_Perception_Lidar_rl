"""Build case-balanced retention data from measured 50498 stair rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent/'mujoco_validation/deploy/mujoco_v135'))


def build(checkpoint, baseline, output):
    from policy import SavedPolicy
    import torch
    torch.set_num_threads(1)
    baseline = baseline.resolve()
    saved = SavedPolicy(checkpoint)
    observations, actions, cases, sources = [], [], [], []
    paths = []
    for row in json.loads((baseline/'ascent/results.json').read_text()):
        if row['terrain'] != 'ascent':
            continue
        if not row['whole_body_top_supported'] or row['fell'] or row['observed_bypass']:
            raise ValueError('Stair teacher must complete every reference case')
        paths.append((baseline/'ascent'/f"ascent_{row['tread']}_{row['command_speed']}.npz", 'ascent'))
    for row in json.loads((baseline/'descent/results.json').read_text()):
        if not row['goal_reached'] or row['fell'] or row['observed_bypass']:
            raise ValueError('Descent teacher must complete every reference case')
        paths.append((baseline/'descent'/f"descent_{row['tread']}_{row['command_speed']}"/'trajectory.npz', 'descent'))
    for case, (path, scene) in enumerate(paths):
        manifest = path.parent/'manifest.json'
        identity = json.loads(manifest.read_text())
        if identity['checkpoint_sha256'] != saved.metadata['checkpoint_sha256']:
            raise ValueError('Stair replay source mismatch')
        with np.load(path, allow_pickle=False) as d:
            active = d['policy_active'] if scene == 'ascent' else d['mode'] == 'policy'
            mask = active & (d['command'][:, 0] > .1)
            obs = d['observation'][mask].astype(np.float32)
        if not len(obs):
            raise ValueError('Empty stair reference: '+str(path))
        # Recompute deterministic teacher means from PRE-step observations.
        # This avoids recording postprocessed actions or mismatched timestamps.
        target = np.stack([saved(x) for x in obs]).astype(np.float32)
        observations.append(obs); actions.append(target)
        cases.append(np.full(len(obs), case, dtype=np.int64))
        sources.append(dict(scene=scene, trajectory=str(path.relative_to(ROOT)),
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), samples=len(obs)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, observation=np.concatenate(observations),
                        action=np.concatenate(actions), case_id=np.concatenate(cases))
    metadata = dict(protocol='flat_stability_50498_v1', source_sha256=saved.metadata['checkpoint_sha256'],
        replay_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), sources=sources,
        samples=sum(map(len, observations)), scope='All moving ascent/descent frames, including B1; no flat imitation')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(samples=metadata['samples'], cases=len(cases), output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    build(**vars(p.parse_args()))
