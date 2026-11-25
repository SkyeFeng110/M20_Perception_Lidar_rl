"""Frozen-policy flat command diagnostics; no training or robot connection.

Sensitivity variants are hypothetical perturbations, not calibrated hardware
models.  Joint targets, joint angles and wheel paths are reported separately.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent/'mujoco_validation/deploy/mujoco_v135'
sys.path.insert(0, str(BASE))
from policy import SavedPolicy
from run import Robot
from command_filter import forward_slew, FORWARD_ACCELERATION, FORWARD_DECELERATION


def run_case(saved, case):
    args = SimpleNamespace(robot_xml=BASE/'assets/m20/mjcf/M20.xml',
        terrain_xml=BASE/'terrains/flat.xml', physics_dt=.001, stair_height=None,
        tread_depth=None, stair_count=None, stair_start=None, spawn_xy=None,
        wall_alpha=.25, perception='clean', seed=42)
    r = Robot(saved, args)
    for g in range(r.model.ngeom):
        if g in r.terrain_geoms or r.model.geom_bodyid[g] in r.wheels:
            r.model.geom_friction[g, 0] = case.get('friction', 1.)
    # Change simulator torque response only; never modify SavedPolicy or gains
    # shared with another case.
    gain = np.ones(16)
    gain[3:6] = case.get('fr_gain', 1.)
    delay = deque([np.zeros(16) for _ in range(case.get('delay_ticks', 0)+1)],
                  maxlen=case.get('delay_ticks', 0)+1)
    last = np.zeros(16)
    filtered = 0.
    records = {k: [] for k in ('time', 'command', 'observation', 'action', 'q',
        'target', 'wheel_y_body', 'pitch', 'pitch_rate', 'speed', 'yaw', 'position',
        'requested_command', 'processed_action')}
    for _ in range(round(6./saved.period)):
        t = float(r.data.time)
        speed = case.get('speed', 1.) if t >= 2. else 0.
        rise = case.get('rise_s', 0.)
        if rise:
            speed *= np.clip((t-2.)/rise, 0., 1.)
        if case.get('partial_full') and 2. <= t < 3.5:
            speed = .3
        requested = speed
        if case.get('forward_filter', False):
            filtered = float(forward_slew(filtered, requested, saved.period, where=np.where, clip=np.clip))
            speed = filtered
        cmd = np.array([speed, case.get('side', 0.) if t >= 2. else 0.,
                        case.get('yaw', 0.) if t >= 2. else 0.])
        obs = r.observe(cmd, last)
        # Rotate observed gravity for a constant IMU mounting/attitude error;
        # leave true physics untouched. These are sensitivity tests only.
        for axis, key in ((0, 'roll_bias_deg'), (1, 'pitch_bias_deg')):
            angle = np.radians(case.get(key, 0.))
            rot = np.eye(3)
            i, j = ((1, 2) if axis == 0 else (0, 2))
            rot[i, i] = rot[j, j] = np.cos(angle)
            rot[i, j] = -np.sin(angle); rot[j, i] = np.sin(angle)
            obs[3:6] = rot@obs[3:6]
        if speed > .05 and 'feature' in case:
            obs[57:63] = case['feature']
        raw_action = saved(obs)
        action = saved.executed_action(raw_action, cmd)
        if case.get('idle_wheel_stop', False) and np.linalg.norm(cmd) < .08:
            action[12:] = 0.
        delay.append(action.copy())
        applied = delay[0]
        if t >= 1.:
            last = saved.history_action(action)
        for _ in range(round(saved.period/.001)):
            if t >= 1.:
                r.control(applied)
                r.data.ctrl[r.aids] *= gain
            else:
                r.control_pose(saved.nominal)
            mujoco.mj_step(r.model, r.data)
        mujoco.mj_forward(r.model, r.data)
        rotation = r.data.xmat[r.base_id].reshape(3, 3)
        gravity = rotation.T@np.array([0., 0., -1.])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 r.base_id, velocity, 0)
        relative = (r.data.xpos[r.wheels]-r.data.xpos[r.base_id])@rotation
        values = dict(time=t, command=cmd, observation=obs, action=raw_action,
            processed_action=action, requested_command=requested,
            q=r.data.qpos[r.qids].copy(), target=saved.position_target(action),
            wheel_y_body=relative[:, 1], position=r.data.xpos[r.base_id].copy(),
            pitch=np.degrees(np.arctan2(gravity[0], np.linalg.norm(gravity[1:]))),
            pitch_rate=np.degrees((rotation.T@velocity[:3])[1]), speed=velocity[3],
            yaw=np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        for k, value in values.items():
            records[k].append(np.copy(value))
    x = {k: np.asarray(v) for k, v in records.items()}
    moving = x['time'] >= 2.
    first = int(np.flatnonzero(x['time'] >= 2.)[0])
    transient = moving & (x['time'] < 3.5)
    result = dict(case=case, pitch_peak_deg=float(np.max(np.abs(x['pitch'][moving]))),
        transient_pitch_peak_deg=float(np.max(np.abs(x['pitch'][transient]))),
        pitch_rate_peak_deg_s=float(np.max(np.abs(x['pitch_rate'][moving]))),
        front_hipx_angle_peak_deg=np.degrees(np.max(np.abs(x['q'][moving][:, [0, 3]]), axis=0)).tolist(),
        front_hipx_target_peak_deg=np.degrees(np.max(np.abs(x['target'][moving][:, [0, 3]]), axis=0)).tolist(),
        fr_inward_y_change_m=float(np.max(x['wheel_y_body'][moving, 1]-x['wheel_y_body'][first-1, 1])),
        front_wheel_action_step=(x['action'][first, 12:14]-x['action'][first-1, 12:14]).tolist(),
        front_hipx_target_step_deg=np.degrees(x['target'][first, [0, 3]]-x['target'][first-1, [0, 3]]).tolist(),
        lateral_peak_m=float(np.max(np.abs(x['position'][moving, 1]-x['position'][first-1, 1]))),
        yaw_peak_deg=float(np.max(np.abs(np.degrees(np.unwrap(np.radians(x['yaw'][moving])))-x['yaw'][first-1]))),
        final_speed_m_s=float(np.mean(x['speed'][-50:])),
        base_min_height_m=float(x['position'][moving, 2].min()))
    result.update(front_width_min_m=float(np.min(x['wheel_y_body'][moving, 0]-x['wheel_y_body'][moving, 1])),
        hipx_target_jump_peak_deg=float(np.degrees(np.abs(np.diff(x['target'][:, [0, 3]], axis=0))[moving[1:]].max())),
        sagittal_target_jump_peak_deg=float(np.degrees(np.abs(np.diff(x['target'][:, [1, 2, 4, 5]], axis=0))[moving[1:]].max())),
        wheel_target_jump_peak_rad_s=float(np.abs(np.diff(x['target'][:, 12:], axis=0))[moving[1:]].max()))
    return result, x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sensitivity', action='store_true')
    p.add_argument('--benchmark', action='store_true', help='Three raw + three production-filtered cases, with deployment stop/history')
    p.add_argument('--no-trajectories', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    saved = SavedPolicy(a.checkpoint)
    a.output.mkdir(parents=True, exist_ok=False)
    cases = [dict(name='keyboard_07_ramp1', speed=.7, rise_s=1.),
             dict(name='light_step03', speed=.3), dict(name='step07', speed=.7),
             dict(name='full_step10', speed=1.), dict(name='full_ramp06', speed=1., rise_s=.6),
             dict(name='full_ramp1', speed=1., rise_s=1.),
             dict(name='partial_to_full', speed=1., partial_full=True),
             dict(name='full_step_idle_stop', speed=1., idle_wheel_stop=True)]
    if a.sensitivity:
        cases += [dict(name='full_step_delay20ms', delay_ticks=1),
                  dict(name='full_step_delay40ms', delay_ticks=2),
                  dict(name='full_step_fr_gain085', fr_gain=.85),
                  dict(name='full_step_friction06', friction=.6),
                  dict(name='full_step_roll_bias2', roll_bias_deg=2.),
                  dict(name='full_step_pitch_bias2', pitch_bias_deg=2.),
                  dict(name='full_step_yaw005', yaw=.05),
                  dict(name='full_step_side005', side=.05),
                  dict(name='full_step_false_edge', feature=[1., .16, .5, 0., 0., 0.])]
    if a.benchmark:
        if a.sensitivity:
            p.error('--benchmark and --sensitivity are separate protocols')
        cases = []
        for filtered in (False, True):
            for name, options in (('light', dict(speed=.3)), ('full', dict(speed=1.)),
                                  ('partial_full', dict(speed=1., partial_full=True))):
                cases.append(dict(name=('filtered_' if filtered else 'raw_')+name,
                    forward_filter=filtered, idle_wheel_stop=True, **options))
    (a.output/'manifest.json').write_text(json.dumps(dict(saved.metadata,
        physics_enabled=True, policy_updated=False, robot_connection=False,
        duration=6., command_onset=2., policy_start=1., physics_dt=.001,
        startup='nominal servo 1s, policy idle 1s',
        protocol='flat_start_v1' if a.benchmark else 'flat_command_diagnostic_v1',
        forward_acceleration_m_s2=FORWARD_ACCELERATION, forward_deceleration_m_s2=FORWARD_DECELERATION,
        deployment_idle_stop=a.benchmark,
        scope='Frozen policy diagnostic; perturbations are hypothetical, not hardware identification'), indent=2)+'\n')
    rows = []
    for case in cases:
        row, trajectory = run_case(saved, case)
        rows.append(row)
        if not a.no_trajectories:
            np.savez_compressed(a.output/(case['name']+'.npz'), **trajectory)
        print(json.dumps(row), flush=True)
    (a.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
