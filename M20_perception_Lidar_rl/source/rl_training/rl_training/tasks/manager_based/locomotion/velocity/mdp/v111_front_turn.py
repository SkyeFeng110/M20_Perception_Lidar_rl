"""Flat-only grounded turning, front initiation and rear touchdown control."""
import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from rl_training.front_turn import turn_terms, front_follow_step, touchdown_risk
from rl_training.traction_transition import rolling_residual
from .v135_terrain_geometry import column_names


def state(env):
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    if not hasattr(env, '_m20_front_turn'):
        z = lambda *shape: torch.zeros((env.num_envs, *shape), device=env.device)
        legs = ('fl', 'fr', 'hl', 'hr')
        env._m20_front_turn = dict(step=-1,
            wheels=[robot.find_bodies(l+'_wheel')[0][0] for l in legs],
            hips=[robot.find_bodies(l+'_hipy')[0][0] for l in legs],
            contacts=[sensor.find_bodies(l+'_wheel')[0][0] for l in legs],
            columns=torch.tensor([n == 'flat' for n in column_names(
                env.cfg.scene.terrain.terrain_generator)], device=env.device),
            ready_time=z(), front_started=z().bool(), rear_released=z(2).bool(),
            previous_yaw=z(), previous_speed=z(4), previous_force=z(4), released=z(4),
            home_height=z()+.56, length=env.episode_length_buf.clone(),
            pending=torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
            samples=z(), sums=z(8), peaks=z(2))
    s = env._m20_front_turn
    if s['step'] == int(env.common_step_counter):
        return s
    cmd = env.command_manager.get_command('base_velocity')
    flat = s['columns'][env.scene.terrain.terrain_types]
    active = flat & (cmd[:, 2].abs() >= .08) & (cmd[:, 1].abs() < .05)
    reset = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    onset = reset | (cmd[:, 2]*s['previous_yaw'] <= 0) | ((cmd[:, 2].abs() >= .55) & (s['previous_yaw'].abs() < .55))
    pos = robot.data.body_link_pos_w[:, s['wheels']]
    linear = robot.data.body_link_lin_vel_w[:, s['wheels']]
    angular = robot.data.body_link_ang_vel_w[:, s['wheels']]
    raw_force = sensor.data.net_forces_w[:, s['contacts']]
    force = raw_force[..., 2].clamp_min(0)
    force_peak = sensor.data.net_forces_w_history[:, :, s['contacts'], 2].amax(1).clamp_min(0)
    height = (pos[..., 2]-float(env.cfg.v11_perception.wheel_radius)
              -env.scene.env_origins[:, None, 2]).clamp_min(0)
    clearance = robot.data.root_pos_w[:, 2]-env.scene.env_origins[:, 2]
    # Capture a bounded quiet standing reference; never integrate crouching
    # during a turn into a progressively lower height target.
    home = reset | (flat & (cmd.abs().amax(1) < .08) & (force >= 15.).all(1)
                    & (robot.data.root_lin_vel_w[:, 2].abs() < .08))
    s['home_height'] = torch.where(home, clearance.clamp(.54, .58), s['home_height'])
    relative = pos-robot.data.root_pos_w[:, None]
    moment = (relative[..., 0]*raw_force[..., 1]-relative[..., 1]*raw_force[..., 0])*cmd[:, 2].sign()[:, None]
    positive = moment.clamp_min(0)
    front, rear = positive[:, :2].sum(1), positive[:, 2:].sum(1)
    follow = front_follow_step(s['ready_time'], s['front_started'], s['rear_released'],
        force, height, front, onset, active, float(env.step_dt))
    s.update({key:follow[key] for key in ('ready_time', 'front_started', 'rear_released')})
    q = robot.data.root_quat_w[:, None].expand(-1, 4, -1)
    base_omega = robot.data.root_ang_vel_w[:, None].expand(-1, 4, -1)
    rel_vel = quat_apply_inverse(q, linear-robot.data.root_lin_vel_w[:, None]
                                -torch.linalg.cross(base_omega, relative, dim=-1))
    reach = quat_apply_inverse(q, pos-robot.data.body_link_pos_w[:, s['hips']])[..., 0]
    s.update(turn_terms(cmd, robot.data.root_lin_vel_b, robot.data.root_ang_vel_b[:, 2],
        robot.data.projected_gravity_b, clearance, s['home_height'], force, height, rel_vel,
        reach, follow['rear_unled'], front, rear, active & ~reset))
    # A full control frame unloaded re-arms touchdown. A force threshold
    # crossing without that release is not counted as a new footstep.
    touchdown = (force >= 15.) & (s['released'] >= .02-1e-6) & ~reset[:, None]
    risk = touchdown_risk(force_peak, s['previous_force'], linear[..., 2], s['previous_speed'], touchdown)
    s['impact'] = risk*active/float(env.step_dt)
    s['released'] = torch.where((force < 15.) & ~reset[:, None], s['released']+float(env.step_dt), 0.)
    residual = rolling_residual(linear, angular, float(env.cfg.v11_perception.wheel_radius)).norm(dim=-1)
    s['slip'] = (residual*(force >= 15.)).sum(1)/4.
    s['active'] = active
    s['back_speed'] = (-robot.data.root_lin_vel_b[:, 0]).clamp_min(0)
    for key in ('samples', 'sums', 'peaks'):
        s[key][reset] = 0
    valid = active & ~reset
    values = torch.stack((s['yaw_error'], s['back_speed'], s['drift'], (force >= 15.).float().mean(1),
                          s['front_share'], s['rear_unled'], risk, s['slip']), 1)
    s['samples'] += valid
    s['sums'] += values*valid[:, None]
    peaks = torch.stack((s['back_speed'], force_peak[:, 2:].amax(1)), 1)
    s['peaks'] = torch.maximum(s['peaks'], peaks*valid[:, None])
    s.update(step=int(env.common_step_counter), previous_yaw=cmd[:, 2].clone(),
        previous_speed=linear[..., 2].clone(), previous_force=force.clone(),
        length=env.episode_length_buf.clone())
    s['pending'][:] = False
    return s


def cost(env):
    return state(env)['cost']*float(getattr(env, '_m20_v2_turn_multiplier', 1.))


def quality(env):
    return state(env)['quality']


def impact(env):
    return -state(env)['impact']


class TurnAwareOriginal(ManagerTermBase):
    """Keep original rewards and exact entity resolution outside flat turns."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Isaac Lab resolves only top-level SceneEntityCfg parameters. These
        # used to be top-level before wrapping; resolve the same objects once
        # here so a hip-only penalty never accidentally selects all joints.
        for value in cfg.params['reward_params'].values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)

    def __call__(self, env, reward_func, reward_params):
        if isinstance(reward_func, str):
            from isaaclab.utils import string_to_callable
            reward_func = string_to_callable(reward_func)
        return reward_func(env, **reward_params)*~state(env)['active']


def rolling_slip(env, command_name, sensor_cfg, asset_cfg=SceneEntityCfg('robot'),
                 cmd_lin_threshold=.1, cmd_ang_threshold=.1):
    from .rewards import feet_slide_ang_z_cmd
    s = state(env)
    # The original penalizes wheel-centre motion relative to the body and
    # therefore taxes ordinary yaw rolling. Use bottom-point residual here.
    original = feet_slide_ang_z_cmd(env, command_name=command_name, sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg, cmd_lin_threshold=cmd_lin_threshold, cmd_ang_threshold=cmd_ang_threshold)
    return torch.where(s['active'], s['slip'], original)


def metric(env, env_ids, key):
    s = state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if key == 'samples':
        return s['samples'][ids].sum()
    if key in ('back_speed_peak', 'rear_force_peak'):
        k = ('back_speed_peak', 'rear_force_peak').index(key)
        return s['peaks'][ids, k].max() if len(ids) else s['samples'].new_zeros(())
    keys = ('yaw_error', 'back_speed', 'drift', 'contact_fraction', 'front_share', 'rear_unled', 'impact', 'slip')
    return s['sums'][ids, keys.index(key)].sum()/s['samples'][ids].sum().clamp_min(1)
