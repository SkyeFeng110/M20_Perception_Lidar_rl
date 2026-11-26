"""Fixed left/right flat turns, with measured backward drift and touchdown.

This is a physics benchmark for training-time selection, not an acoustic dBA
measurement. Importing this module does not start MuJoCo or run a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'm20_front_led_grounded_turn_v1'
CASES = tuple((vx, yaw*sign, mu) for vx, yaw, mu in
              ((0., .2, 1.), (0., .6, 1.), (0., .4, .6), (.3, .4, 1.)) for sign in (1., -1.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return tuple(row[k] for k in ('forward', 'yaw', 'friction'))


def wheel_loads(robot):
    """Actual terrain-contact force in world coordinates, including yaw arm."""
    import mujoco
    force = np.zeros((4, 3)); moment = np.zeros(4); local = np.zeros(6)
    for i, contact in enumerate(robot.data.contact):
        g1, g2 = map(int, contact.geom)
        if g1 in robot.terrain_geoms:
            wheel, sign = g2, 1.
        elif g2 in robot.terrain_geoms:
            wheel, sign = g1, -1.
        else:
            continue
        indices = np.flatnonzero(robot.wheels == robot.model.geom_bodyid[wheel])
        if not len(indices):
            continue
        mujoco.mj_contactForce(robot.model, robot.data, i, local)
        world = sign*np.asarray(contact.frame).reshape(3, 3).T@local[:3]
        arm = contact.pos-robot.data.xpos[robot.base_id]
        index = indices[0]
        force[index] += world
        moment[index] += arm[0]*world[1]-arm[1]*world[0]
    return force, moment


def summarize(a, vx, yaw, mu, fell):
    t = a['time']; moving = t >= 2.; steady = t >= 3.
    mask = moving if moving.any() else np.ones(len(t), dtype=bool)
    stable = steady if steady.any() else mask
    vel = a['velocity'][mask]
    support = a['force'][mask] >= 15.
    height = a['height'][mask]
    turn_ratio = a['yaw_rate'][stable]*np.sign(yaw)/abs(yaw)
    positive = np.maximum(a['moment'][mask]*np.sign(yaw), 0.)
    total = positive.sum(1)
    useful = total > 2.
    rear_air = (height[:, 2:] > .012) & ~support[:, 2:]
    events = a['touchdown'][mask]
    return dict(forward=vx, yaw=yaw, friction=mu, fell=bool(fell), samples=int(mask.sum()),
        moving_seconds=float(mask.sum()*.02),
        yaw_tracking_ratio=float(np.mean(turn_ratio)),
        yaw_error_mean=float(np.abs(a['yaw_rate'][stable]-yaw).mean()),
        backward_distance_m=float(np.maximum(-vel[:, 0], 0.).sum()*.02),
        backward_speed_peak_m_s=float(np.maximum(-vel[:, 0], 0.).max()),
        pure_xy_speed_mean=float(np.linalg.norm(vel[:, :2], axis=1).mean()),
        forward_error_mean=float(np.abs(vel[:, 0]-vx).mean()),
        contact_fraction=float(support.mean()),
        rear_lift_fraction=float(rear_air.mean()),
        rear_unled_fraction=float(a['rear_unled'][mask].mean()),
        rear_touchdown_count=int(events[:, 2:].sum()),
        rear_force_peak_n=float(a['force_peak'][mask, 2:].max()),
        touchdown_down_speed_peak_m_s=float(a['touch_speed'][mask].max()),
        front_positive_moment_share=float((positive[useful, :2].sum(1)/total[useful]).mean()) if useful.any() else 0.,
        pitch_peak_deg=float(np.abs(a['pitch'][mask]).max()),
        roll_peak_deg=float(np.abs(a['roll'][mask]).max()))


def validate(path, checkpoint):
    path = Path(path)
    m = json.loads((path.parent/'manifest.json').read_text())
    rows = json.loads(path.read_text())
    if (m.get('protocol') != PROTOCOL or not m.get('physics_enabled')
            or m.get('checkpoint_sha256') != sha(checkpoint)
            or m.get('results_sha256') != sha(path) or m.get('duration') != 10.
            or m.get('period') != .02 or m.get('physics_dt') != .001
            or len(rows) != len(CASES) or {identity(r) for r in rows} != set(CASES)):
        raise ValueError('Invalid front-turn benchmark identity or fixed cases')
    if m.get('actor_dimensions') != [63, 16]:
        raise ValueError('Front-turn benchmark actor ABI changed')
    required = ('samples', 'yaw_tracking_ratio', 'yaw_error_mean', 'backward_distance_m',
        'backward_speed_peak_m_s', 'pure_xy_speed_mean', 'forward_error_mean', 'contact_fraction',
        'rear_lift_fraction', 'rear_unled_fraction', 'rear_touchdown_count', 'rear_force_peak_n',
        'touchdown_down_speed_peak_m_s', 'front_positive_moment_share', 'pitch_peak_deg', 'roll_peak_deg')
    for row in rows:
        if not isinstance(row.get('fell'), bool) or row.get('samples', 0) < 1:
            raise ValueError('Invalid front-turn physical result')
        if any(not isinstance(row.get(k), (int, float)) or not np.isfinite(row[k]) for k in required):
            raise ValueError('Missing or invalid measured turn metric')
        for key, value in row.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError('Nonfinite turn metric: '+key)
    return rows


def run(a):
    if (a.output/'results.json').is_file() and (a.output/'manifest.json').is_file():
        validate(a.output/'results.json', a.checkpoint)
        print('[M20] Reusing matching measured front-turn baseline', flush=True)
        return
    import mujoco
    import torch
    sys.path.insert(0, str(a.validation_root.resolve()/'deploy/mujoco_v135'))
    from policy import SavedPolicy
    from run import Robot
    torch.set_num_threads(1)
    saved = SavedPolicy(a.checkpoint, a.config_dir)
    if saved.period != .02:
        raise ValueError('Front-turn benchmark requires the native 20ms period')
    base = a.validation_root.resolve()/'deploy/mujoco_v135'
    destination = a.output.resolve()
    token = uuid.uuid4().hex[:12]
    a.output = destination.with_name(destination.name+'.partial_'+token)
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, (vx, yaw, mu) in enumerate(CASES):
        args = SimpleNamespace(robot_xml=base/'assets/m20/mjcf/M20.xml', terrain_xml=base/'terrains/flat.xml',
            physics_dt=.001, stair_height=None, tread_depth=None, stair_count=None, stair_start=None,
            spawn_xy=None, wall_alpha=.25, perception='clean', seed=42)
        r = Robot(saved, args)
        for geom in range(r.model.ngeom):
            if geom in r.terrain_geoms or r.model.geom_bodyid[geom] in r.wheels:
                r.model.geom_friction[geom, 0] = mu
        records = {k:[] for k in ('time', 'velocity', 'yaw_rate', 'height', 'force', 'force_peak',
            'moment', 'touchdown', 'touch_speed', 'pitch', 'roll', 'rear_unled')}
        last = np.zeros(16); previous_vz = np.zeros(4); released = np.zeros(4); fell = False
        front_started = False; front_ready = 0.; rear_released = False
        for _ in range(500):
            t = float(r.data.time); active = t >= 1.
            cmd = np.asarray([vx, 0., yaw] if t >= 2. else [0., 0., 0.])
            action = saved.executed_action(saved(r.observe(cmd, last)), cmd)
            peak = np.zeros(4)
            downward = np.maximum(-previous_vz, 0.)
            # Force/velocity peaks use physics substeps, not only a possibly
            # already-rebounded 20ms policy frame.
            for _ in range(20):
                r.control(action) if active else r.control_pose(saved.nominal)
                mujoco.mj_step(r.model, r.data)
                forces, _ = wheel_loads(r)
                peak = np.maximum(peak, forces[:, 2])
                for i, body in enumerate(r.wheels):
                    speed = np.zeros(6)
                    mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_XBODY, int(body), speed, 0)
                    downward[i] = max(downward[i], -speed[5])
            mujoco.mj_forward(r.model, r.data)
            last = saved.history_action(action) if active else np.zeros(16)
            rotation = r.data.xmat[r.base_id].reshape(3, 3)
            gravity = rotation.T@np.asarray([0., 0., -1.])
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_XBODY, r.base_id, vel, 0)
            wheel_velocity = np.zeros((4, 6))
            for i, body in enumerate(r.wheels):
                mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_XBODY, int(body), wheel_velocity[i], 0)
            ground_force, moment = wheel_loads(r)
            force = np.maximum(ground_force[:, 2], 0.)
            peak = np.maximum(peak, force)
            touchdown = (force >= 15.) & (released >= .02-1e-6)
            released = np.where(force < 15., released+.02, 0.)
            previous_vz = wheel_velocity[:, 5].copy()
            height = np.maximum(r.data.xpos[r.wheels, 2]-.09, 0.)
            rear_air = (height[2:] > .012) & (force[2:] < 15.)
            if t >= 2.:
                front_started |= bool(((height[:2] > .012) & (force[:2] < 15.)).any()
                    or np.maximum(moment[:2]*np.sign(yaw), 0.).sum() > 1.)
                front_ready = front_ready+.02 if front_started and (force[:2] >= 15.).all() else 0.
                rear_released |= bool(rear_air.any())
                unled = rear_air & (front_ready < .08-1e-6)
                if rear_released and (force[2:] >= 15.).all():
                    front_started = False; front_ready = 0.; rear_released = False
            else:
                unled = np.zeros(2, dtype=bool)
            values = dict(time=t, velocity=rotation.T@vel[3:], yaw_rate=(rotation.T@vel[:3])[2],
                height=height, force=force, force_peak=peak, rear_unled=unled,
                moment=moment,
                touchdown=touchdown, touch_speed=downward*touchdown,
                pitch=np.degrees(np.arctan2(gravity[0], np.linalg.norm(gravity[1:]))),
                roll=np.degrees(np.arctan2(-gravity[1], -gravity[2])))
            for key, value in values.items():
                records[key].append(np.copy(value))
            if not np.isfinite(r.data.qpos).all() or not np.isfinite(r.data.qvel).all():
                raise RuntimeError('Nonfinite turn simulation')
            if active and (rotation[2, 2] < .35 or r.data.xpos[r.base_id, 2]-r.ground() < .22):
                fell = True
                break
        arrays = {k:np.asarray(v) for k, v in records.items()}
        np.savez_compressed(a.output/f'case_{case}.npz', **arrays)
        row = summarize(arrays, vx, yaw, mu, fell)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    manifest = dict(protocol=PROTOCOL, checkpoint_sha256=sha(a.checkpoint), physics_enabled=True,
        actor_dimensions=[63, 16], period=.02, physics_dt=.001, duration=10., seed=42,
        results_sha256=sha(a.output/'results.json'),
        robot_xml_sha256=sha(base/'assets/m20/mjcf/M20.xml'), terrain_xml_sha256=sha(base/'terrains/flat.xml'),
        startup='standing pose to 1s, policy idle to 2s, raw turn to 10s',
        force_method='mj_contactForce at actual terrain contacts in world frame; substep peaks',
        checkpoint_metadata=saved.metadata)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    validate(a.output/'results.json', a.checkpoint)
    # Publish only complete identity-checked suites. Interrupted work stays
    # available for diagnosis and never poisons the reusable baseline cache.
    if destination.exists():
        if (destination/'results.json').is_file() and (destination/'manifest.json').is_file():
            validate(destination/'results.json', a.checkpoint)
            return
        destination.rename(destination.with_name(destination.name+'.incomplete_'+token))
    a.output.rename(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config-dir', type=Path)
    p.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
