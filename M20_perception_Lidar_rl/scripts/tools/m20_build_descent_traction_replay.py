"""Add terrain labels to the immutable 58098 replay; no new physics rollouts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/rl_training/rl_training'))
from descent_traction import PROTOCOL
from early_traction import sha


def build(source, teacher, output):
    source, teacher, output = [Path(p).resolve() for p in (source, teacher, output)]
    meta = json.loads(source.with_suffix('.json').read_text())
    expected = dict(protocol=PROTOCOL, teacher_sha256=sha(teacher), source_replay_sha256=sha(source),
                    source_metadata_sha256=sha(source.with_suffix('.json')))
    if meta.get('sha256') != expected['source_replay_sha256'] or meta.get('teacher_sha256') != expected['teacher_sha256']:
        raise ValueError('V2 replay must be derived from its measured 58098 teacher')
    if output.is_file() and output.with_suffix('.json').is_file():
        old = json.loads(output.with_suffix('.json').read_text())
        if any(old.get(k) != v for k, v in expected.items()) or old.get('sha256') != sha(output):
            raise ValueError('Existing V2 replay differs; choose a new output instead of overwriting it')
        return old
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Incomplete V2 replay output exists; preserve it and choose a new output')
    with np.load(source, allow_pickle=False) as d:
        packed = {k: d[k].copy() for k in d.files}
    scope = np.full(len(packed['case']), -1, np.int64)
    for case, origin in enumerate(meta['sources']):
        for k in ('trajectory', 'manifest'):
            if sha(origin[k]) != origin[k+'_sha256']:
                raise ValueError('Measured teacher source changed: '+origin[k])
        scope[packed['case'] == case] = ('flat', 'ascent', 'descent').index(origin['scope'])
    if set(scope) != {0, 1, 2}:
        raise ValueError('V2 requires measured flat, ascent and descent coverage')
    packed['scope'] = scope
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    expected.update(sha256=sha(output), source_replay=str(source), samples=len(scope),
        scope_samples={n:int((scope == i).sum()) for i, n in enumerate(('flat', 'ascent', 'descent'))},
        scope_cases={n:len(np.unique(packed['case'][scope == i])) for i, n in enumerate(('flat', 'ascent', 'descent'))})
    output.with_suffix('.json').write_text(json.dumps(expected, indent=2)+'\n')
    return expected


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'teacher', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
