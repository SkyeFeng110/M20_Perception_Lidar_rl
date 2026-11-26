"""Compare full stair completion and long flat behavior against 50498."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def stair_regressions(ascent, descent, reference_ascent, reference_descent):
    failures = []
    def keyed(rows):
        return {(r['terrain'], r['tread'], r['command_speed']): r for r in rows if r['terrain'] != 'flat'}
    for rows, originals in ((ascent, reference_ascent), (descent, reference_descent)):
        current, before = keyed(rows), keyed(originals)
        if current.keys() != before.keys():
            failures.append('missing or extra stair cases')
            continue
        for key, b in before.items():
            r = current[key]
            if r['fell'] or r['observed_bypass']:
                failures.append(f'{key}: fell or bypass')
            finish = 'whole_body_top_supported' if key[0] == 'ascent' else 'goal_reached'
            if b[finish] and not r[finish]:
                failures.append(f'{key}: lost complete flight')
            if r['simulated_seconds'] > b['simulated_seconds']*1.20+.5:
                failures.append(f'{key}: traversal slowed')
            if key[0] == 'ascent':
                speed = r['later_stairs_route_velocity_x']
                if (r['stall_detected'] or speed is None or not math.isfinite(speed)
                        or speed < .9*b['later_stairs_route_velocity_x']):
                    failures.append(f'{key}: lost forward progress')
                for field, tolerance in (('forward_yaw_max_change_deg', 2.), ('forward_lateral_max_abs', .10)):
                    if r[field] > b[field]+tolerance:
                        failures.append(f'{key}: {field}')
                for field in ('skipped_treads', 'rear_skipped_treads'):
                    if any(len(a) > len(old) for a, old in zip(r[field], b[field])):
                        failures.append(f'{key}: more skipped treads')
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'baseline', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    for name, script, flags in (
        ('ascent', 'm20_entry_flat_benchmark.py', ['--speeds', '.5', '.7', '1.0', '--no-trajectories']),
        ('descent', 'm20_quiet_supplement.py', []),
        ('baseline_long', 'm20_flat_command_probe.py', ['--benchmark', '--duration', '20', '--no-trajectories'])):
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/tools'/script), '--checkpoint', str(a.checkpoint),
                            '--output', str(a.output/name), *flags], check=True, env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=600)
    read = lambda root, name: json.loads((root/name/'results.json').read_text())
    failures = stair_regressions(read(a.output, 'ascent'), read(a.output, 'descent'),
                                read(a.baseline, 'ascent'), read(a.baseline, 'descent'))
    flat = read(a.output, 'baseline_long')
    # Raw full-stick is mandatory. Filtering cannot hide a failed transient.
    flat_passed = all(r['steady_reach_p95_m'] <= .12 and r['pitch_peak_deg'] <= 3.
        and r['pitch_rate_peak_deg_s'] <= 20. and r['base_min_height_m'] >= .40
        and r['final_speed_m_s'] >= .8*r['case']['speed']
        and r['lateral_peak_m'] <= .15 and r['yaw_peak_deg'] <= 3. for r in flat)
    result = dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        stair_retention_passed=not failures, regressions=failures, long_flat_passed=flat_passed,
        deployable_candidate=not failures and flat_passed, flat_cases=flat,
        scope='Fixed 6 ascent + 4 descent cases; completion retention is not every-tread certification')
    (a.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'flat_cases'}))
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
