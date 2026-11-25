"""Preserve measured raw-command flat starts without copying steering targets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = 'm20_route_hold_56898_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(teacher, flat, output):
    teacher, flat, output = map(lambda p: Path(p).resolve(), (teacher, flat, output))
    manifest = flat/'manifest.json'
    m = json.loads(manifest.read_text())
    if (m.get('checkpoint_sha256') != sha(teacher) or not m.get('physics_enabled')
            or m.get('protocol') != 'flat_start_v1' or m.get('policy_updated') is not False):
        raise ValueError('Expected measured flat-start teacher evidence')
    observations, actions, sources = [], [], []
    for case in ('raw_light', 'raw_full', 'raw_partial_full'):
        path = flat/(case+'.npz')
        with np.load(path, allow_pickle=False) as d:
            # Flat probe C starts at one second. Retain startup and subsequent
            # movement, using actions actually executed under target bounds.
            keep = d['time'] >= 1.
            index = np.flatnonzero(keep)
            observations.append(d['observation'][keep]); actions.append(d['processed_action'][keep])
        sources.append(dict(trajectory=str(path), trajectory_sha256=sha(path),
            manifest=str(manifest), manifest_sha256=sha(manifest),
            source_index=index.tolist(), samples=len(index)))
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite route preservation evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, observation=np.concatenate(observations), action=np.concatenate(actions))
    meta = dict(protocol=PROTOCOL, sha256=sha(output), teacher_sha256=sha(teacher),
        samples=sum(s['samples'] for s in sources), sources=sources,
        scope='Measured raw flat startup and motion; sagittal joints and common drive only')
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    return dict(samples=meta['samples'], no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'flat', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
