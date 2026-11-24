"""Read-only diagnosis of fixed evaluations/replay; no model updates or PPO."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(row):
    return (row['profile'], row['terrain'], row['tread'], row['command_speed'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--evaluation', type=int, required=True)
    p.add_argument('--initial-results', type=Path, required=True)
    p.add_argument('--trajectory-evaluation', type=int, default=1000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    target = a.session/f'eval_{a.evaluation:06d}'
    rows = json.loads((target/'results.json').read_text())
    baseline = {key(r): r for r in json.loads(a.initial_results.read_text())}
    fields = ['whole_body_top_supported', 'all_treads_supported', 'fell', 'stall_detected',
        'first_supported_tread', 'skipped_treads', 'after_c_idle_yaw_max_change_deg',
        'forward_yaw_max_change_deg', 'forward_lateral_max_abs', 'p95_max_abs_reach',
        'route_mean_velocity_x', 'later_stairs_route_velocity_x']
    report = dict(evaluation=a.evaluation, source_results=str(target/'results.json'),
        initial_results=str(a.initial_results), cases=[], no_training_started=True)
    for row in rows:
        report['cases'].append(dict(case=list(key(row)),
            initial={k:baseline[key(row)].get(k) for k in fields},
            current={k:row.get(k) for k in fields}))
    history = []
    for path in sorted(a.session.glob('eval_*/results.json')):
        update = int(path.parent.name.split('_')[1])
        if update > a.evaluation:
            continue
        ascent = [r for r in json.loads(path.read_text()) if r['terrain'] == 'ascent']
        history.append(dict(updates=update, top=sum(r['whole_body_top_supported'] for r in ascent),
            full_tread_support=sum(r['all_treads_supported'] for r in ascent)))
    report['support_history'] = history
    state_path = json.loads((target/'acceptance.json').read_text())['candidates'][0]['checkpoint']
    sys.path.insert(0, str(ROOT.parent/'mujoco_validation/deploy/mujoco_v135'))
    from policy import SavedPolicy
    policy = SavedPolicy(state_path)
    native = np.load(ROOT/'data/m20_recovery/reference_observations_v2.npz')
    obs = torch.from_numpy(native['observation'])
    phase = torch.from_numpy(native['phase'])
    m = load('plateau_targeted_math', SOURCE/'targeted_reference.py')
    with torch.inference_mode():
        means = policy.actor(obs)
        reflected = policy.actor(m.mirror_observation(obs))
        gap = (means-m.mirror_action(reflected)).square().mean(1).sqrt()
    report['deployment_replay'] = dict(samples=len(obs),
        settled_idle_samples=int(m.settled_idle_mask(obs, phase).sum()),
        mirror_action_rms_raw_by_phase={str(int(ph)):float(gap[phase == ph].mean()) for ph in phase.unique()},
        phases_are_used_for_sampling_in_old_profile_but_not_anchor_mask=True)
    root = a.session/f'eval_{a.trajectory_evaluation:06d}'
    z = np.load(root/'keyboard/flat_None_0.7.npz')
    idle = z['policy_active'] & (z['phase'] == 1)
    idle_actions = z['action'][idle]
    wheels = idle_actions[:,12:].mean(0)
    report['takeover_trajectory'] = dict(evaluation=a.trajectory_evaluation,
        idle_samples=int(idle.sum()), zero_command_max=float(abs(z['command'][idle]).max()),
        wheel_mean_raw=wheels.tolist(), wheel_left_minus_right_raw=float(wheels[[0,2]].mean()-wheels[[1,3]].mean()),
        hipx_mean_raw=idle_actions[:,[0,3,6,9]].mean(0).tolist(),
        initial_observation=z['observation'][idle][0].tolist(),
        angular_velocity_rms_rad_s=np.sqrt((z['angular_velocity'][idle]**2).mean(0)).tolist())
    report['short_or_missing_support'] = []
    for path in sorted((root/'standing').glob('ascent*.npz')):
        z = np.load(path); labels = z['supported_tread'][:,:2]
        short = []
        for side in range(2):
            maximum = {t:0 for t in range(1,13)}
            start = 0
            while start < len(labels):
                end = start+1
                while end < len(labels) and labels[end,side] == labels[start,side]:
                    end += 1
                tread = int(labels[start,side])
                if tread in maximum:
                    maximum[tread] = max(maximum[tread], end-start)
                start = end
            short.append({str(t):frames for t,frames in maximum.items() if frames < 3})
        report['short_or_missing_support'].append(dict(case=path.stem, maximum_run_frames_below_three=short))
    report['temporal_credit'] = dict(policy_dt_s=.02, gamma=.99, gae_lambda=.95,
        gae_effective_steps=1/(1-.99*.95), gae_effective_seconds=.02/(1-.99*.95),
        support_delay_s=.06, tread_transit_seconds_at_point7={'.26':.26/.7, '.30':.30/.7},
        scope='Discount/GAE horizon is a diagnostic, not a proven unique cause')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output), evaluations=len(history),
        latest_support=history[-1], replay=report['deployment_replay']), indent=2))


if __name__ == '__main__':
    main()
