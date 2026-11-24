"""Supplementary descent/noise cases; separate from the fixed 12-case gate.

Falls and incomplete descents are recorded as behavior results, not exceptions.
Invalid outputs or simulation failures remain runtime errors.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def validate_result(row):
    if row.get('status') == 'runtime_error' or not isinstance(row.get('goal_reached'), bool):
        raise RuntimeError('Invalid supplementary descent result')
    proxies = row.get('acoustic_proxies', {})
    moving = proxies.get('phases', {}).get('moving', {})
    seconds = moving.get('seconds')
    if (proxies.get('schema_version') != 1 or not isinstance(seconds, (float, int))
            or not math.isfinite(seconds) or seconds <= 0):
        raise RuntimeError('Supplementary descent has no valid moving acoustic samples')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = a.validation_root.resolve()/'deploy/mujoco_v135'
    rows = []
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for tread in (.26, .30):
        for speed in (.5, .7):
            target = (a.output/f'descent_{tread}_{speed}').resolve()
            command = [sys.executable, str(base/'run.py'), '--headless', '--checkpoint', str(a.checkpoint.resolve()),
                       '--terrain-xml', str(base/'terrains/stairs_descent_walls.xml'),
                       '--stair-height', '.16', '--tread-depth', str(tread), '--command', str(speed), '0', '0',
                       '--duration', '25', '--stop-at-goal', '--output', str(target)]
            with target.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=cpu_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            result = json.loads((target/'summary.json').read_text())
            validate_result(result)
            rows.append(dict(case=f'descent/{tread}/{speed}', terrain='descent', tread=tread,
                             command_speed=speed, **result))
            (a.output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(supplementary=True, cases=len(rows),
                         completed=sum(row['goal_reached'] for row in rows),
                         falls=sum(row['fell'] for row in rows), output=str(a.output))), flush=True)


if __name__ == '__main__':
    main()
