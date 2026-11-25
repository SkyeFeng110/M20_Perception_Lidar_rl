"""Original-route reward on flat ground and ascent, including first-riser transfer."""
import torch
from rl_training.route_hold import latch_reference, route_terms


def reset(env, env_ids):
    from .v111_flat_posture import reset as parent_reset
    parent_reset(env, env_ids)
    if hasattr(env, '_m20_route_hold'):
        env._m20_route_hold['pending'][env_ids] = True


def state(env):
    from .v135_terrain_geometry import column_names, ascent_name
    robot = env.scene['robot']
    if not hasattr(env, '_m20_route_hold'):
        names = column_names(env.cfg.scene.terrain.terrain_generator)
        n, d = env.num_envs, env.device
        zeros = lambda *shape: torch.zeros((n, *shape), device=d)
        env._m20_route_hold = dict(step=-1,
            flat_columns=torch.tensor([x == 'flat' for x in names], device=d),
            ascent_columns=torch.tensor([ascent_name(x) for x in names], device=d),
            heading=zeros(), origin=zeros(2), valid=zeros().bool(), pending=torch.ones(n, device=d, dtype=torch.bool),
            length=env.episode_length_buf.clone(), samples=zeros(2), sums=zeros(2, 5), peaks=zeros(2, 4))
    s = env._m20_route_hold
    if s['step'] == int(env.common_step_counter):
        return s
    reset_mask = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    command = env.command_manager.get_command('base_velocity')
    term = env.command_manager.get_term('base_velocity')
    terrain = env.scene.terrain.terrain_types
    flat, ascent = s['flat_columns'][terrain], s['ascent_columns'][terrain]
    s['focused'] = flat | ascent
    straight = (s['focused'] & (command[:, 0] >= 0)
                & (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03) & ~term.is_heading_env)
    s['heading'], s['origin'], s['valid'] = latch_reference(
        s['heading'], s['origin'], s['valid'], robot.data.heading_w,
        robot.data.root_pos_w[:, :2], straight, reset_mask)
    heading = s['heading']
    delta = robot.data.root_pos_w[:, :2]-s['origin']
    yaw = robot.data.heading_w-heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    lateral = -delta[:, 0]*heading.sin()+delta[:, 1]*heading.cos()
    velocity = robot.data.root_lin_vel_w
    speed = velocity[:, 0]*heading.cos()+velocity[:, 1]*heading.sin()
    side = -velocity[:, 0]*heading.sin()+velocity[:, 1]*heading.cos()
    s.update(route_terms(yaw, lateral, side, robot.data.root_ang_vel_w[:, 2],
                         speed, command[:, 0], straight, ascent))
    for key in ('samples', 'sums', 'peaks'):
        s[key][reset_mask] = 0
    values = torch.stack([s[k] for k in ('yaw', 'lateral', 'lateral_speed', 'yaw_rate', 'speed')], 1)
    for i, terrain_mask in enumerate((flat, ascent)):
        active = s['active'] & terrain_mask & ~reset_mask
        s['samples'][:, i] += active
        s['sums'][:, i] += values*active[:, None]
        s['peaks'][:, i] = torch.maximum(s['peaks'][:, i], values[:, :4]*active[:, None])
    s.update(step=int(env.common_step_counter), length=env.episode_length_buf.clone())
    s['pending'][:] = False
    return s


def cost(env):
    return state(env)['cost']


def quality(env):
    return state(env)['quality']


def flat_posture_cost(env):
    from .v111_flat_posture import state as posture_state
    s = posture_state(env)
    # Keep stance, pitch and onset protection; its old resampled route cost
    # must not oppose a correction toward our persistent original route.
    return -(1.2*s['shape']+.6*s['attitude'])*s['active']-s['onset_cost']


def flat_posture_quality(env):
    from .v111_flat_posture import state as posture_state
    s = posture_state(env)
    command = env.command_manager.get_command('base_velocity')[:, 0]
    progress = (state(env)['speed']/command.clamp_min(.1)).clamp(0, 1)
    return progress*(.6*(1-s['shape'])+.25*(1-s['attitude'])+.15)*s['active']


def outside_focus_reward(env, reward_func, reward_params):
    # One route objective owns focused flat/ascent surfaces. Descents and
    # other terrains retain their inherited reward behavior.
    if isinstance(reward_func, str):
        # Hydra can serialize nested callable parameters as module:name.
        from isaaclab.utils import string_to_callable
        reward_func = string_to_callable(reward_func)
    return reward_func(env, **reward_params)*~state(env)['focused']


def metric(env, env_ids, key):
    s = state(env)
    terrain, quantity = key.split('_', 1)
    i = ('flat', 'ascent').index(terrain)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if quantity == 'samples':
        return s['samples'][ids, i].sum()
    fields = ('yaw', 'lateral', 'lateral_speed', 'yaw_rate', 'speed')
    if quantity.endswith('_peak'):
        k = fields.index(quantity[:-5])
        return s['peaks'][ids, i, k].max() if len(ids) else s['samples'].new_zeros(())
    k = fields.index(quantity.removesuffix('_sum'))
    total = s['sums'][ids, i, k].sum()
    return total if quantity.endswith('_sum') else total/s['samples'][ids, i].sum().clamp_min(1)
