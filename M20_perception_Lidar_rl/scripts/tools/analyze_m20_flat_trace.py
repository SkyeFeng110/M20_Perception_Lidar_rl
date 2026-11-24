"""Read-only real-controller trace metrics; never connect to a robot."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    footer = [r for r in rows if r.get('event') == 'capture_end']
    complete = len(footer) == 1 and footer[0].get('complete') is True
    if not complete:
        raise ValueError('Trace is incomplete; require capture_end complete=true')
    inference = [r for r in rows if r.get('event') == 'policy_inference'
                 and r.get('observation_built') and r.get('inferred')]
    if not inference: raise ValueError('No measured policy inference rows')
    obs = np.asarray([r['obs63'] for r in inference], dtype=float)
    if obs.shape[1:] != (63,) or not np.isfinite(obs).all():
        raise ValueError('Invalid trace observations')
    t = np.asarray([r['mono_s'] for r in inference])
    episode = np.asarray([r['policy_episode'] for r in inference])
    same = episode[1:] == episode[:-1]
    delta = np.diff(t)[same]
    moving = obs[:, 6] > .1
    if not moving.any(): raise ValueError('No forward-command interval')
    gravity = obs[:, 3:6]
    pitch = np.degrees(np.arctan2(gravity[:, 0], np.linalg.norm(gravity[:, 1:], axis=1)))
    # q/default/scale evidence is stored in robot order. Reconstruct the
    # measured native joints from the actual observation contract per row.
    hipx, targets, measured = [], [], []
    for r, x in zip(inference, obs):
        mapping = np.asarray(r['policy_index_for_robot'], dtype=int)
        default = np.asarray(r['default_position_robot'], dtype=float)
        q = default+x[9:25][mapping]
        target = np.asarray(r['intended_command_16x5'], dtype=float).reshape(16, 5)[:, 1]
        hipx.append(q[[0, 4]])
        targets.append(target); measured.append(q)
    target = np.asarray(targets)
    requested = [r.get('requested_velocity3') for r in inference]
    valid_request = all(v is not None and len(v) == 3 for v in requested)
    valid_request = valid_request and bool(np.isfinite(np.asarray(requested, dtype=float)).all())
    output = dict(scope='measured_controller_trace; commanded speed is not measured base speed',
        complete=True, samples=len(inference), request_available=valid_request,
        commanded_forward_max_m_s=float(obs[:, 6].max()),
        pitch_peak_deg=float(np.abs(pitch[moving]).max()),
        pitch_rate_peak_deg_s=float(np.degrees(np.abs(obs[moving, 1]/.25)).max()),
        front_hipx_peak_deg=np.degrees(np.max(np.abs(np.asarray(hipx)[moving]), axis=0)).tolist(),
        policy_interval_p50_ms=float(np.median(delta)*1000) if len(delta) else None,
        policy_interval_p99_ms=float(np.quantile(delta, .99)*1000) if len(delta) else None,
        front_target_jump_peak_deg=float(np.degrees(np.abs(np.diff(target[:, [0, 1, 2, 4, 5, 6]], axis=0))[same].max())) if same.any() else None,
        target_to_current_front_error_peak_deg=float(np.degrees(np.abs(target-np.asarray(measured))[moving][:, [0, 1, 2, 4, 5, 6]].max())),
        calibration_note='Joint error compares a newly issued target with current feedback, not a measured motor delay; pitch is the onboard gravity estimate.')
    if valid_request:
        output['requested_forward_max_m_s'] = float(np.asarray(requested)[:, 0].max())
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.trace)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
