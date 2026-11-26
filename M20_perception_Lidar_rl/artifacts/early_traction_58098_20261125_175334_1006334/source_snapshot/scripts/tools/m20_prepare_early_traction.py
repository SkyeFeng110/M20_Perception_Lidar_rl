"""Record immutable 58098 retention data before the user-started PPO stage."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/rl_training/rl_training'))
from early_traction import PROTOCOL, sha
from m20_build_flat_posture_replay import build_posture
from m20_build_route_flat_replay import build as build_flat


def preserve_incomplete(path):
    if path.exists():
        path.rename(path.with_name(path.name+f'.incomplete_{time.time_ns()}'))


def collect(tool, checkpoint, output, extra=()):
    if (output/'results.json').is_file():
        # Completed manifests are checked again by preflight below. An
        # interrupted suite cannot masquerade as a complete cache.
        rows = json.loads((output/'results.json').read_text())
        expected = {'native': 12, 'entry': 3, 'flat_start': 6, 'production': 9, 'descent': 4}[output.name]
        if len(rows) == expected:
            return
    preserve_incomplete(output)
    command = [sys.executable, str(ROOT/'scripts/tools'/tool), '--checkpoint', str(checkpoint),
               '--output', str(output), *extra]
    print('[58098 retention] collecting '+output.name+'; log: '+str(output.with_suffix('.log')), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)


def build_protection(reference, data, teacher):
    arrays = {k: [] for k in ('observation', 'action', 'early_wheel', 'case')}
    sources = []
    meta = json.loads((data/'current_gait.json').read_text())
    start = float(meta['stair_start'])

    def add(trajectory, manifest, observation, action, early, index, scope):
        if len(index) < 10:
            return
        manifest_data = json.loads(manifest.read_text())
        if manifest_data.get('checkpoint_sha256') != sha(teacher) or not manifest_data.get('physics_enabled'):
            raise ValueError('Protection trajectory has the wrong teacher: '+str(trajectory))
        arrays['observation'].append(observation[index].astype(np.float32))
        arrays['action'].append(action[index].astype(np.float32))
        arrays['early_wheel'].append(early[index].astype(bool))
        arrays['case'].append(np.full(len(index), len(sources), np.int64))
        sources.append(dict(trajectory=str(trajectory), trajectory_sha256=sha(trajectory),
            manifest=str(manifest), manifest_sha256=sha(manifest), scope=scope,
            samples=len(index), source_index=index.tolist()))

    # Include native commanded speeds plus actual raw full-stick startup.
    suites = [reference/'native/standing', reference/'native/keyboard',
              reference/'production/current', reference/'production/flat-start']
    for suite in suites:
        for row in json.loads((suite/'results.json').read_text()):
            if row['fell'] or row['observed_bypass'] or row['stall_detected']:
                continue
            path = suite/f"{row['terrain']}_{row['tread']}_{row['command_speed']}.npz"
            with np.load(path, allow_pickle=False) as d:
                early = np.zeros((len(d['time']), 4), bool)
                if row['terrain'] == 'ascent':
                    x = d['wheel_position'][..., 0]
                    early = ((x[:, :2].max(1)[:, None] >= start-.11) & (x < start+3.*row['tread']+.09)
                             & (d['command'][:, :1] > .1))
                index = np.flatnonzero(d['policy_active'] & np.isin(d['phase'], [1, 2, 3]))
                add(path, suite/'manifest.json', d['observation'], d['action'], early, index, row['terrain'])
    for name in ('raw_light', 'raw_full', 'raw_partial_full'):
        path = reference/'flat_start'/(name+'.npz')
        with np.load(path, allow_pickle=False) as d:
            add(path, path.parent/'manifest.json', d['observation'], d['processed_action'],
                np.zeros((len(d['time']), 4), bool), np.flatnonzero(d['time'] >= 1.), 'flat')
    for row in json.loads((reference/'descent/results.json').read_text()):
        if row['fell'] or row['observed_bypass'] or not row['goal_reached']:
            continue
        path = reference/'descent'/f"descent_{row['tread']}_{row['command_speed']}"/'trajectory.npz'
        with np.load(path, allow_pickle=False) as d:
            add(path, path.parent/'manifest.json', d['observation'], d['action'],
                np.zeros((len(d['time']), 4), bool), np.flatnonzero(d['mode'] == 'policy'), 'descent')
    if {s['scope'] for s in sources} != {'flat', 'ascent', 'descent'}:
        raise ValueError('Need successful measured flat/ascent/descent protection; inspect baseline logs')
    output = data/'protected_58098.npz'
    np.savez_compressed(output, **{k: np.concatenate(v) for k, v in arrays.items()})
    output.with_suffix('.json').write_text(json.dumps(dict(protocol=PROTOCOL, sha256=sha(output),
        teacher_sha256=sha(teacher), descent_results_sha256=sha(reference/'descent/results.json'),
        samples=sum(s['samples'] for s in sources), sources=sources), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'data'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    checkpoint, reference, data = [p.resolve() for p in (a.checkpoint, a.reference, a.data)]
    if sha(checkpoint) != 'c21b28919e97e17c99a48776c9189b0fb57e1427318b0d94dfb7ec2cd8cfa0db':
        raise ValueError('This repair baseline must be the selected original model_58098.pt')
    ready = reference/'ready.json'
    if ready.is_file():
        saved = json.loads(ready.read_text())
        if saved.get('protocol') != PROTOCOL or any(sha(k) != v for k, v in saved['files'].items()):
            raise ValueError('Prepared retention data changed; use new reference/data directories')
        print('[58098 retention] reusing verified frozen reference and replay', flush=True)
        return
    reference.mkdir(parents=True, exist_ok=True)
    teacher = reference/'teacher/model_58098.pt'
    teacher.parent.mkdir(exist_ok=True)
    for source, dest in [(checkpoint, teacher), *[(checkpoint.parent/'params'/name, teacher.parent/'params'/name)
                                                for name in ('env.yaml', 'agent.yaml')]]:
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            if sha(dest) != sha(source):
                raise ValueError('Frozen teacher/config mismatch: '+str(dest))
        else:
            shutil.copy2(source, dest)
    collect('m20_repair_acceptance.py', teacher, reference/'native', ['--baseline-only'])
    collect('m20_entry_flat_benchmark.py', teacher, reference/'entry',
            ['--startup', 'z', '--keyboard-profile', '--keyboard-idle-seconds', '0', '--speeds', '.7'])
    collect('m20_flat_command_probe.py', teacher, reference/'flat_start', ['--benchmark'])
    collect('m20_production_transition_benchmark.py', teacher, reference/'production')
    collect('m20_quiet_supplement.py', teacher, reference/'descent', ['--retention-metrics'])
    if not (data/'current_gait.npz').is_file() or not (data/'compact_flat.json').is_file():
        preserve_incomplete(data)
        build_posture(reference/'native/results.json', teacher,
            reference/'production/current/flat_None_1.0.npz', teacher, data, preserve_current_flat=True)
    if not (data/'route_flat.npz').is_file():
        build_flat(teacher, reference/'flat_start', data/'route_flat.npz')
    build_protection(reference, data, teacher)
    files = [p for root in (reference, data) for p in root.rglob('*')
             if p.is_file() and '.incomplete_' not in str(p) and p.suffix in ('.npz', '.json', '.yaml', '.pt')]
    ready.write_text(json.dumps(dict(protocol=PROTOCOL, files={str(p): sha(p) for p in files},
                                    no_training_started=True), indent=2)+'\n')
    print('[58098 retention] baseline and action protection ready; no PPO updates performed', flush=True)


if __name__ == '__main__':
    main()
