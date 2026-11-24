"""Retain later ascent without cloning the defective front B1 transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from m20_build_recovery_replay import build


def build_entry(baseline_dir, checkpoint, output, *, after_four_b2=False):
    metadata = build(baseline_dir, checkpoint, output)
    with np.load(output, allow_pickle=False) as data:
        packed = {key: data[key].copy() for key in data.files}
    masks = []
    for case in metadata['cases']:
        with np.load(case['trajectory'], allow_pickle=False) as d:
            keep = d['policy_active'] & np.isin(d['phase'], [1, 2, 3])
            mask = np.zeros(len(d['time']), dtype=bool)
            if case['terrain'] == 'ascent':
                ready = []
                for side in range(4 if after_four_b2 else 2):
                    ready.append(next((i+2 for i in range(len(mask)-2)
                        if ((d['supported_tread'][i:i+3, side] >= 2).all() if after_four_b2
                            else (d['supported_tread'][i:i+3, side] == 1).all())), None))
                if not all(i is not None for i in ready):
                    if not after_four_b2:
                        raise ValueError('Cannot release B1 without measured front B1 support')
                else:
                    mask = d['time'] >= d['time'][max(ready)]+.12
            masks.append(mask[keep])
            case['post_entry_samples'] = int(mask[keep].sum())
    packed['post_entry'] = np.concatenate(masks)
    np.savez_compressed(output, **packed)
    metadata['entry_mask_protocol'] = ('measured_four_wheel_b2_plus_120ms_v1' if after_four_b2
                                     else 'measured_bilateral_b1_plus_120ms_v1')
    metadata['replay_sha256'] = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    Path(output).with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return dict(samples=metadata['samples'], post_entry_samples=int(packed['post_entry'].sum()),
                source_sha256=metadata['source_sha256'], no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline-dir', 'checkpoint', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--after-four-b2', action='store_true')
    a = p.parse_args()
    print(json.dumps(build_entry(a.baseline_dir, a.checkpoint, a.output,
                                after_four_b2=a.after_four_b2), indent=2))
