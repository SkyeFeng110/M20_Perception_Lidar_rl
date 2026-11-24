"""Bounded-memory mechanical quietness diagnostics; no audio or dBA estimate."""
import numpy as np
import mujoco


def contact_point_velocity(linear, angular, radius):
    return linear+np.cross(angular, radius)


class Moments:
    def __init__(self):
        self.n = 0
        self.squares = 0.
        self.maximum = 0.

    def add(self, values):
        x = np.asarray(values, dtype=float)
        if not np.isfinite(x).all():
            raise RuntimeError('Nonfinite acoustic proxy')
        if x.size:
            self.n += x.size
            self.squares += float(np.square(x).sum())
            self.maximum = max(self.maximum, float(np.abs(x).max()))

    def rms(self):
        return (self.squares/self.n)**.5 if self.n else None


class AcousticRecorder:
    def __init__(self):
        self.groups = {}
        self.reset()

    def reset(self):
        for group in self.groups.values():
            group['previous_pos'] = None
        self.last_phase = None
        self.contact = None
        self.air = np.zeros(4)
        self.window = np.zeros(4)
        self.previous_force = np.zeros(4)
        self.previous_velocity = np.zeros((4, 6))
        self.previous_torque = None
        self.previous_action = self.previous_delta = None

    def group(self, moving):
        key = 'moving' if moving else 'idle'
        if key not in self.groups:
            self.groups[key] = dict(seconds=0., touchdowns=0, distance=0., previous_pos=None,
                **{k: Moments() for k in ('approach', 'load_rate', 'slip', 'leg_speed', 'wheel_speed',
                                          'torque', 'torque_rate', 'leg_reversal', 'wheel_reversal')})
        return self.groups[key]

    def policy_sample(self, action, *, active, moving):
        if not active:
            self.reset()
            return
        action = np.asarray(action)
        if self.previous_action is not None:
            delta = action-self.previous_action
            if self.previous_delta is not None:
                reversal = np.minimum(np.abs(delta), np.abs(self.previous_delta))*(delta*self.previous_delta < 0)
                g = self.group(moving)
                g['leg_reversal'].add(reversal[:12])
                g['wheel_reversal'].add(reversal[12:])
            self.previous_delta = delta.copy()
        self.previous_action = action.copy()

    def physics_sample(self, robot, *, active, moving):
        if not active:
            return
        m, d = robot.model, robot.data
        dt = float(m.opt.timestep)
        g = self.group(moving)
        if self.last_phase != moving:
            g['previous_pos'] = None
            self.last_phase = moving
        force = np.zeros(4)
        approach = np.zeros(4)
        slip = []
        velocity = np.zeros((4, 6))
        for i, body in enumerate(robot.wheels):
            # XBODY is centered at the body frame (xpos), matching the radius
            # below; BODY would refer to the inertial frame instead.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(body), velocity[i], 0)
        for ci, contact in enumerate(d.contact):
            a, b = map(int, contact.geom)
            rg = b if a in robot.terrain_geoms else a if b in robot.terrain_geoms else -1
            if rg < 0:
                continue
            ids = np.flatnonzero(robot.wheels == m.geom_bodyid[rg])
            if not len(ids):
                continue
            i = int(ids[0])
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, wrench)
            normal_force = max(0., float(wrench[0]))
            if normal_force <= 0:
                continue
            normal = contact.frame[:3]*(1. if rg == b else -1.)
            radius = contact.pos-d.xpos[robot.wheels[i]]
            point = contact_point_velocity(velocity[i, 3:], velocity[i, :3], radius)
            before = contact_point_velocity(self.previous_velocity[i, 3:], self.previous_velocity[i, :3], radius)
            force[i] += normal_force
            approach[i] = max(approach[i], -float(point@normal), -float(before@normal))
            slip.append(np.linalg.norm(point-normal*(point@normal)))
        if self.contact is None:
            self.contact = force >= 15.
            self.previous_force = force.copy()
            self.previous_velocity = velocity.copy()
        contact = np.where(self.contact, force > 5., force >= 15.)
        touch = contact & ~self.contact & (self.air >= .06)
        self.window = np.where(touch, .06, np.maximum(0., self.window-dt))
        g['touchdowns'] += int(touch.sum())
        g['approach'].add(approach[touch])
        g['load_rate'].add(np.maximum(0., force-self.previous_force)[(self.window > 0) & contact]/dt)
        g['slip'].add(slip)
        g['leg_speed'].add(d.qvel[robot.vids[:12]])
        g['wheel_speed'].add(d.qvel[robot.vids[12:]])
        torque = d.ctrl[robot.aids].copy()
        g['torque'].add(torque)
        if self.previous_torque is not None:
            g['torque_rate'].add((torque-self.previous_torque)/dt)
        position = d.xpos[robot.base_id, :2].copy()
        if g['previous_pos'] is not None:
            g['distance'] += float(np.linalg.norm(position-g['previous_pos']))
        g['previous_pos'] = position
        g['seconds'] += dt
        self.air = np.where(contact, 0., self.air+dt)
        self.contact, self.previous_force = contact, force.copy()
        self.previous_velocity, self.previous_torque = velocity, torque

    def report(self):
        result = dict(schema_version=1, evidence='mechanical_proxies_only_no_audio_or_dBA',
                      physics_sampling=True, phases={})
        for key, g in self.groups.items():
            result['phases'][key] = dict(seconds=g['seconds'], path_distance_m=g['distance'],
                touchdowns=g['touchdowns'],
                touchdown_approach_rms_m_s=g['approach'].rms(),
                touchdown_approach_max_m_s=g['approach'].maximum if g['approach'].n else None,
                touchdown_load_rate_rms_N_s=g['load_rate'].rms(),
                touchdown_load_rate_max_N_s=g['load_rate'].maximum if g['load_rate'].n else None,
                contact_point_slip_rms_m_s=g['slip'].rms(),
                leg_joint_speed_rms_rad_s=g['leg_speed'].rms(),
                wheel_joint_speed_rms_rad_s=g['wheel_speed'].rms(),
                torque_rms_Nm=g['torque'].rms(), torque_rate_rms_Nm_s=g['torque_rate'].rms(),
                leg_action_reversal_rms=g['leg_reversal'].rms(),
                wheel_action_reversal_rms=g['wheel_reversal'].rms())
        return result
