"""B1-only support transfer; retain the native gait on subsequent stairs."""
import os
import torch
from rl_training.entry_transfer import make_state, transfer_step
from .v111_step_support import StepSupportVelocityCommand


class EntrySequenceVelocityCommand(StepSupportVelocityCommand):
    """Raw deployment speeds on ascent, including direct 0 -> 1 m/s."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from .v135_terrain_geometry import column_names
        self._flat_sequence_columns = torch.tensor([n == 'flat' for n in column_names(
            env.cfg.scene.terrain.terrain_generator)], device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        from .v135_terrain_geometry import v135_terrain_geometry
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[v135_terrain_geometry(self._env)['ascent'][ids]]
        self.refine_speed[ids] = self.vel_command_b.new_tensor([.5, .7, 1.])[
            torch.randint(3, (len(ids),), device=self.device)]
        self.vel_command_b[ids] = 0
        direct = ids[~self.refine_cohort[ids]]
        self.vel_command_b[direct, 0] = self.refine_speed[direct]
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False
        all_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        stress = (self._flat_sequence_columns[self._env.scene.terrain.terrain_types[all_ids]]
                  & (all_ids.remainder(8) >= 1) & (all_ids.remainder(8) <= 3))
        chosen = all_ids[stress]
        self.vel_command_b[chosen] = 0
        self.is_heading_env[chosen] = False
        self.is_standing_env[chosen] = False

    def _update_command(self):
        super()._update_command()
        from .v135_terrain_geometry import v135_terrain_geometry
        ids = v135_terrain_geometry(self._env)['ascent'] & self.refine_cohort
        self.vel_command_b[ids, 0] = self.refine_speed[ids]*(self.refine_elapsed[ids] >= 2.)
        self.vel_command_b[ids, 1:] = 0
        indices = torch.arange(self.num_envs, device=self.device)
        profile = indices.remainder(8)
        flat = (self._flat_sequence_columns[self._env.scene.terrain.terrain_types]
                & (profile >= 1) & (profile <= 3))
        elapsed = self.refine_elapsed
        speed = torch.where((profile == 3) | ((profile == 2) & (elapsed < 2.)), .3, 1.)
        self.vel_command_b[flat, 0] = speed[flat]*(elapsed[flat] >= 1.)
        self.vel_command_b[flat, 1:] = 0


def after_front_entry_reward(env, reward_func, reward_params):
    """Keep supplemental B2+ objectives from paying for an unfinished B1."""
    value = reward_func(env, **reward_params)
    transfer = state(env)['transfer']
    return value*(~transfer['eligible'] | transfer['finished'])


def after_rear_approach_reward(env, reward_func, reward_params):
    """Diagonal mirror regularization must not ask a distant rear leg to lift."""
    value = reward_func(env, **reward_params)
    return value*~state(env)['rear_wait']


def flat_motion_state(env):
    from isaaclab.utils.math import quat_apply_inverse
    from .v135_terrain_geometry import column_names
    from rl_training.entry_sequence import flat_motion_terms
    if not hasattr(env, '_m20_sequence_flat'):
        robot = env.scene['robot']
        legs = ('fl', 'fr', 'hl', 'hr')
        env._m20_sequence_flat = dict(step=-1,
            wheels=[robot.find_bodies(leg+'_wheel')[0][0] for leg in legs],
            hips=[robot.find_bodies(leg+'_hipy')[0][0] for leg in legs],
            columns=torch.tensor([n == 'flat' for n in column_names(
                env.cfg.scene.terrain.terrain_generator)], device=env.device))
    s = env._m20_sequence_flat
    if s['step'] == int(env.common_step_counter):
        return s
    robot = env.scene['robot']
    pos = robot.data.body_link_pos_w[:, s['wheels']]
    hips = robot.data.body_link_pos_w[:, s['hips']]
    q = robot.data.root_quat_w[:, None].expand(-1, 4, -1)
    reach = quat_apply_inverse(q, pos-hips)[..., 0]
    wheel_y = quat_apply_inverse(q, pos-robot.data.root_pos_w[:, None])[..., 1]
    term = env.command_manager.get_term('base_velocity')
    command = env.command_manager.get_command('base_velocity')
    yaw = robot.data.heading_w-term.refine_heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    offset = robot.data.root_pos_w[:, :2]-term.refine_origin
    lateral = -offset[:, 0]*term.refine_heading.sin()+offset[:, 1]*term.refine_heading.cos()
    speed = (robot.data.root_lin_vel_w[:, 0]*term.refine_heading.cos()
             +robot.data.root_lin_vel_w[:, 1]*term.refine_heading.sin())
    pitch = torch.atan2(robot.data.projected_gravity_b[:, 0],
                       torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=1))
    active = (s['columns'][env.scene.terrain.terrain_types] & ~term.is_heading_env
              & (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03))
    s.update(flat_motion_terms(reach, yaw, lateral, speed, command[:, 0], pitch,
                              robot.data.root_ang_vel_b[:, 1], wheel_y, active))
    s['step'] = int(env.common_step_counter)
    return s


def flat_motion_cost(env):
    return -flat_motion_state(env)['cost']


def flat_motion_quality(env):
    return flat_motion_state(env)['quality']


def flat_motion_metric(env, env_ids, key):
    s = flat_motion_state(env)
    mask = s['active'][env_ids]
    return (s[key][env_ids]*mask).sum()/mask.sum().clamp_min(1)


def reset(env, env_ids):
    from .v111_step_support import reset as parent_reset
    parent_reset(env, env_ids)
    if hasattr(env, '_m20_entry_transfer'):
        env._m20_entry_transfer['pending'][env_ids] = True


def state(env):
    from .v111_step_support import state as support_state
    from .v135_terrain_geometry import v135_terrain_geometry
    support_state(env)
    step = int(env.common_step_counter)
    if not hasattr(env, '_m20_entry_transfer'):
        env._m20_entry_transfer = dict(step=-1, length=env.episode_length_buf.clone(),
            pending=torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
            transfer=make_state(env.num_envs, env.device))
    s = env._m20_entry_transfer
    if s['step'] == step:
        return s
    reset_ids = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    robot, entry = env.scene['robot'], env._v111_first_tread
    ids = env._v111_recovery_wheels[:2]
    pos = robot.data.body_link_pos_w[:, ids]
    vel = robot.data.body_link_lin_vel_w[:, ids]
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts[:2], 2]
    rear = robot.data.body_link_pos_w[:, env._v111_recovery_wheels[2:]]
    rear_x = ((rear[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    rear_force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts[2:], 2]
    x = ((pos[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    command = env.command_manager.get_command('base_velocity')
    geometry = v135_terrain_geometry(env)
    # Independent of the old entry.done flag, which becomes true on the
    # completion tick. The transfer owns its B1 certificate and exits there.
    active = geometry['ascent'] & entry['height_valid'] & (command[:, 0] > .1)
    active &= (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03)
    radius = float(env.cfg.v11_perception.wheel_radius)
    pitch = torch.atan2(robot.data.projected_gravity_b[:, 0],
                       torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=1))
    s.update(transfer_step(s['transfer'], x=x, bottom=pos[..., 2]-radius,
        force=force, wheel_vz=vel[..., 2],
        root_speed=(robot.data.root_lin_vel_w[:, :2]*entry['forward']).sum(-1),
        front_speed=(vel[..., :2]*entry['forward'][:, None]).sum(-1),
        pitch=pitch, pitch_rate=robot.data.root_ang_vel_b[:, 1],
        lower=entry['lower'], rise=entry['height'], tread=entry['tread'],
        active=active, reset=reset_ids, dt=float(env.step_dt),
        rear_x=rear_x, rear_bottom=rear[..., 2]-radius, rear_force=rear_force,
        strict=getattr(env.cfg, 'v111_strict_entry_sequence', False)
               or os.environ.get('M20_ENTRY_DIRECT_REPAIR') == '1'))
    s.update(step=step, length=env.episode_length_buf.clone())
    s['pending'][:] = False
    return s


def progress_reward(env):
    return state(env)['credit']/float(env.step_dt)


def safety_cost(env):
    return -state(env)['cost']/float(env.step_dt)


def sequence_reward(env):
    """Reward the second front transfer only after the first is loaded."""
    return state(env)['sequence_credit']/float(env.step_dt)


def metric(env, env_ids, key):
    current = state(env)
    transfer = current['transfer']
    eligible = transfer['eligible'][env_ids]
    if key == 'eligible_episodes':
        return eligible.sum().float()
    if key == 'clean_complete':
        value = transfer['clean_complete'] & ~transfer['rear_early'].any(1)
    elif key == 'rear_early':
        value = transfer['rear_early'].any(1)
    elif key == 'sequence':
        value = transfer['sequence']
    elif key == 'rear_hold':
        value = transfer['hold_sum']
    elif key in ('first_lift', 'first_support', 'second_legal_lift'):
        side = transfer['first'].clamp_min(0)
        if key == 'second_legal_lift':
            side = 1-side
        evidence = transfer['certified'] if key == 'first_support' else transfer['valid_lift']
        value = evidence.gather(1, side[:, None]).squeeze(1) & (transfer['first'] >= 0)
    else:
        value = transfer[key]
    value = value[env_ids].float()
    if value.ndim > 1:
        value = value.mean(1)
    return (value*eligible).sum()/eligible.sum().clamp_min(1)
