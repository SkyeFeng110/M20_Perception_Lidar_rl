"""Independent native-PD MuJoCo acceleration and loaded-wheel slip benchmark.

This supplements, never replaces, the full entry/flat/descent acceptance.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def command_speed(t, profile, speed):
    if t < 2.:
        return 0.
    if profile == 'hold':
        return speed
    if profile == 'step_release':
        return 1. if t < 6. else .5
    if profile == 'partial_full':
        return .3 if t < 5. else 1.
    if profile == 'ramp':
        return min(1., (t-2.)/.6)
    raise ValueError(profile)


def loaded_contact_slip(robot):
    """Actual contact-point Jacobian, including wheel spin exactly once.

    Only >=15N horizontal terrain contacts. No inferred slip from wheel RPM,
    no COM/link origin ambiguity, no vertical riser or airborne samples.
    """
    import mujoco
    import numpy as np
    total, weight = np.zeros(4), np.zeros(4)
    jac = np.zeros((3, robot.model.nv))
    force = np.zeros(6)
    for i, contact in enumerate(robot.data.contact):
        g1, g2 = map(int, contact.geom)
        terrain, wheel = (g1, g2) if g1 in robot.terrain_geoms else (g2, g1)
        if terrain not in robot.terrain_geoms or wheel < 0:
            continue
        body = int(robot.model.geom_bodyid[wheel])
        indices = np.flatnonzero(robot.wheels == body)
        normal = np.asarray(contact.frame[:3])
        if not len(indices) or abs(normal[2]) < .98:
            continue
        mujoco.mj_contactForce(robot.model, robot.data, i, force)
        if force[0] < 15.:
            continue
        mujoco.mj_jac(robot.model, robot.data, jac, None, contact.pos, body)
        velocity = jac@robot.data.qvel
        tangent = velocity-normal*np.dot(normal, velocity)
        index = indices[0]
        total[index] += np.linalg.norm(tangent)*force[0]
        weight[index] += force[0]
    return np.divide(total, weight, out=np.zeros(4), where=weight > 0), weight > 0


def summarize(arrays):
    import numpy as np
    t, command = arrays['time'], arrays['command']
    moving = command > .1
    # Piecewise target has .8s settling time for tracking; pitch/slip include
    # the complete moving interval, including the acceleration transient.
    change = np.r_[True, np.abs(np.diff(command)) > .01]
    last_change = np.maximum.accumulate(np.where(change, t, -1e6))
    steady = moving & (t-last_change >= .8)
    rear_mask = arrays['support'][:, 2:] & moving[:, None]
    rear_slip = arrays['slip'][:, 2:][rear_mask]
    pitch = np.abs(arrays['pitch'][moving])
    rate = np.abs(arrays['pitch_rate'][moving])
    def percentile(values, q):
        return float(np.percentile(values, q)) if len(values) else None
    windows = []
    for onset in (2., 5., 6.):
        at = int(np.searchsorted(t, onset))
        if at == 0 or at >= len(t) or abs(command[at]-command[at-1]) < .1:
            continue
        mask = (t >= onset) & (t < onset+1.5)
        windows.append(dict(onset=onset, from_speed=float(command[at-1]),
            to_speed=float(command[at]), peak_pitch_deg=percentile(np.abs(arrays['pitch'][mask]), 100),
            peak_pitch_rate_deg_s=percentile(np.abs(arrays['pitch_rate'][mask]), 100),
            front_wheel_action_step_max=float(np.max(np.linalg.norm(
                np.diff(arrays['action'][max(0, at-1):at+3, 12:14], axis=0), axis=1)))
            if len(arrays['action']) >= at+1 else None,
            front_wheel_action_at_onset=arrays['action'][at, 12:14].tolist()
            if len(arrays['action']) > at else None))
    front_tread = arrays.get('front_tread')
    entry_simultaneous = False
    entry_lead_gap = None
    entry_scope = 'from_lower'
    if front_tread is not None and len(front_tread):
        if (front_tread[0] == 1).any():
            # The fixed traction suite spawns some stair cases with the front
            # wheels already on B1. It cannot identify the physical approach
            # to B1 in that case; leave the diagnostic explicitly unset.
            entry_scope = 'starts_on_b1'
        else:
            first = np.full(2, -1, dtype=int)
            first_frame = np.full(2, -1, dtype=int)
            for frame, labels in enumerate(front_tread):
                onset = (labels == 1) & (first < 0)
                if onset.any():
                    sides = np.flatnonzero(onset)
                    if len(sides) == 2:
                        entry_simultaneous = True
                    elif first_frame[1-sides[0]] >= 0 and frame-first_frame[1-sides[0]] <= 2:
                        entry_simultaneous = True
                        entry_lead_gap = frame-first_frame[1-sides[0]]
                    first[onset] = 1
                    first_frame[onset] = frame
    return dict(pitch_peak_deg=percentile(pitch, 100), pitch_p95_deg=percentile(pitch, 95),
        pitch_rate_peak_deg_s=percentile(rate, 100),
        rear_slip_p95=percentile(rear_slip, 95),
        rear_slip_mean=float(rear_slip.mean()) if len(rear_slip) else None,
        rear_loaded_samples=int(rear_mask.sum()),
        rear_support_fraction=float(rear_mask.sum()/max(1, 2*moving.sum())),
        steady_tracking_ratio=float(np.mean(arrays['speed'][steady]/command[steady])) if steady.any() else None,
        progress_m=float(arrays['position'][-1, 0]-arrays['position'][0, 0]),
        lateral_signed_final_m=float(arrays['position'][-1, 1]-arrays['position'][0, 1]),
        lateral_max_abs_m=float(np.max(np.abs(arrays['position'][:, 1]-arrays['position'][0, 1]))),
        yaw_max_abs_deg=float(np.max(np.abs(np.degrees(np.unwrap(np.radians(arrays['yaw'])))-arrays['yaw'][0]))),
        entry_front_simultaneous=(bool(entry_simultaneous) if entry_scope == 'from_lower' else None),
        entry_front_lead_gap_steps=entry_lead_gap,
        entry_front_scope=entry_scope,
        transitions=windows)


def run(a):
    import mujoco
    import numpy as np
    import torch
    sys.path.insert(0, str(a.validation_root.resolve()/'deploy/mujoco_v135'))
    from policy import SavedPolicy
    from run import Robot
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    saved = SavedPolicy(a.checkpoint, a.config_dir)
    base = a.validation_root.resolve()/'deploy/mujoco_v135'
    # 3 flat profiles plus low/mid/full drive on each tread and friction.
    cases = [('flat', None, 'step_release', 1.), ('flat', None, 'partial_full', 1.),
             ('flat', None, 'ramp', 1.)]
    cases += [('ascent', tread, 'hold', speed) for tread in (.26, .30) for speed in (.3, .5, .7, 1.)]
    manifest = dict(saved.metadata, schema_version=1, physics_enabled=True,
        suite='traction_transition_v1', frictions=a.frictions, period=saved.period,
        duration=a.seconds, seed=42, contact_force=15., horizontal_normal_min=.98,
        slip_method='mj_jac at actual loaded terrain contact; tangential velocity in m/s',
        startup='standing pose servo until 1s, policy idle until command begins at 2s',
        scope='supplement; full B1, sequential treads, keyboard and descent acceptance still required')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = []
    for friction in a.frictions:
        for index, (terrain, tread, profile, speed) in enumerate(cases):
            args = SimpleNamespace(robot_xml=base/'assets/m20/mjcf/M20.xml',
                terrain_xml=base/'terrains'/('flat.xml' if terrain == 'flat' else 'stairs_ascent_walls.xml'),
                physics_dt=.001, stair_height=.16 if tread else None, tread_depth=tread,
                stair_count=None, stair_start=None, spawn_xy=None, wall_alpha=.25, perception='clean', seed=42)
            r = Robot(saved, args)
            # MuJoCo combines sliding coefficients with max at equal priority.
            # Set BOTH terrain and wheel coefficients for the low-friction test.
            for geom in range(r.model.ngeom):
                if geom in r.terrain_geoms or r.model.geom_bodyid[geom] in r.wheels:
                    r.model.geom_friction[geom, 0] = friction
            records = {k: [] for k in ('time', 'command', 'pitch', 'pitch_rate', 'position', 'speed', 'yaw', 'slip', 'support', 'front_tread', 'action', 'joint_vel')}
            last = np.zeros(16)
            fell, bypass, top_frames = False, False, 0
            termination = 'duration'
            for _ in range(round(a.seconds/saved.period)):
                t = float(r.data.time)
                active = t >= 1.
                command = command_speed(t, profile, speed)
                cmd = np.array([command, 0., 0.])
                action = saved.executed_action(saved(r.observe(cmd, last)), cmd)
                for _ in range(round(saved.period/.001)):
                    r.control(action) if active else r.control_pose(saved.nominal)
                    mujoco.mj_step(r.model, r.data)
                mujoco.mj_forward(r.model, r.data)
                last = saved.history_action(action) if active else np.zeros(16)
                rotation = r.data.xmat[r.base_id].reshape(3, 3)
                gravity = rotation.T@np.array([0., 0., -1.])
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_XBODY, r.base_id, vel, 0)
                slip, support = loaded_contact_slip(r)
                front_tread = np.full(2, -1, dtype=int)
                if tread:
                    for side in range(2):
                        if support[side]:
                            front_tread[side] = int(round(float(r.data.xpos[r.wheels[side], 2]) / .16))
                values = dict(time=t, command=command, pitch=np.degrees(np.arctan2(gravity[0], np.linalg.norm(gravity[1:]))),
                    pitch_rate=np.degrees((rotation.T@vel[:3])[1]), position=r.data.xpos[r.base_id].copy(),
                    speed=vel[3], yaw=np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])), slip=slip, support=support,
                    front_tread=front_tread, action=action.copy(), joint_vel=r.data.qvel[r.vids].copy())
                for key, value in values.items():
                    records[key].append(value)
                if active and (rotation[2, 2] < .35 or r.data.xpos[r.base_id, 2]-r.ground() < .22):
                    fell = True
                    termination = 'fall'
                    break
                bypass |= (abs(r.data.xpos[r.base_id, 1]) > r.terrain.course_width/2
                           and r.data.xpos[r.base_id, 0] < r.terrain.goal_x)
                if tread:
                    top = int(r.terrain.parameters['step_count'][0])*.16
                    on_top = (support.all() and not bypass and rotation[2, 2] >= .8
                        and (r.data.xpos[r.wheels, 0] > r.terrain.goal_x+saved.perception['wheel_radius']).all()
                        and (np.abs(r.data.xpos[r.wheels, 2]-saved.perception['wheel_radius']-top) <= .035).all()
                        and r.data.xpos[r.base_id, 2]-top >= .30)
                    top_frames = top_frames+1 if on_top else 0
                    if top_frames*saved.period >= .06:
                        termination = 'flight_complete'
                        break
            arrays = {k: np.asarray(v) for k, v in records.items()}
            row = dict(terrain=terrain, tread=tread, profile=profile, speed=speed, friction=friction,
                       fell=fell, bypass=bool(bypass), termination=termination, **summarize(arrays))
            rows.append(row)
            if a.trajectories:
                np.savez_compressed(a.output/f'{friction}_{index}.npz', **arrays)
            print(json.dumps(row, allow_nan=False), flush=True)
    (a.output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config-dir', type=Path)
    p.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    p.add_argument('--frictions', type=float, nargs='+', default=[1., .6])
    p.add_argument('--seconds', type=float, default=14.)
    p.add_argument('--trajectories', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.seconds < 8 or sorted(args.frictions) != [.6, 1.]:
        p.error('Fixed suite requires >=8 seconds and friction coefficients 1.0 and 0.6')
    run(args)
