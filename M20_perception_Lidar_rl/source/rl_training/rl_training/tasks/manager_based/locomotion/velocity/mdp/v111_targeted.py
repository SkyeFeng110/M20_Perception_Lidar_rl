"""Contact-phase landing credit, isolated from the running Refine profile."""
import torch
from rl_training.targeted_reference import settlement_step, ascent_speed_credit


def _landing_reward(env, *, airborne_weight=.5, force_threshold=5.):
    from .v111_refine import state
    from .v135_terrain_geometry import v135_terrain_geometry
    state(env)  # initializes physical geometry and recovery wheel indices
    if not hasattr(env, '_v111_targeted'):
        env._v111_targeted = dict(step=-1, target=torch.zeros((env.num_envs, 2), device=env.device).long(),
            peak=torch.zeros((env.num_envs, 2), device=env.device),
            total=torch.zeros(env.num_envs, device=env.device),
            length=env.episode_length_buf.clone(), pending=torch.zeros(env.num_envs, device=env.device).bool())
    s = env._v111_targeted
    if s['step'] == int(env.common_step_counter):
        return s['credit']/float(env.step_dt)
    reset = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    if s['step'] == -1:
        reset[:] = True
    s['target'][reset], s['peak'][reset] = 0, 0
    s['total'][reset] = 0
    entry, adjacent = env._v111_first_tread, env._v111_adjacent
    geometry = v135_terrain_geometry(env)
    robot = env.scene['robot']
    ids = env._v111_recovery_wheels[:2]
    wheels = robot.data.body_pos_w[:, ids]
    expected = adjacent['overshoot_target']
    x = ((wheels[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    margin = torch.minimum(torch.full_like(entry['tread'], .025), entry['tread']*.1)[:, None]
    start = (expected-1)*entry['tread'][:, None]+margin
    end = expected*entry['tread'][:, None]-margin
    outside = torch.maximum(start-x, x-end).clamp_min(0)
    left = torch.stack((-entry['forward'][:, 1], entry['forward'][:, 0]), 1)
    lateral = ((wheels[..., :2]-env.scene.env_origins[:, None, :2])*left[:, None]).sum(-1)
    width = (left.abs()*geometry['radii']).sum(-1)[:, None]+(expected-1)*entry['tread'][:, None]
    in_course = lateral.abs() <= width-env.cfg.v11_perception.wheel_radius-.02
    alignment = torch.exp(-(outside/.10).square())*in_course*(x <= end)
    bottom = wheels[..., 2]-env.cfg.v11_perception.wheel_radius
    plane = entry['lower'][:, None]+expected*entry['height'][:, None]
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts[:2], 2].clamp_min(0)
    velocity = robot.data.body_link_lin_vel_w[:, ids, 2]
    command = env.command_manager.get_command('base_velocity')
    active = ((geometry['ascent'] & entry['height_valid'] & (command[:, 0] > .1))[:, None]
        & entry['entry_approached'] & (expected >= 1) & (expected <= geometry['flight_riser_count'][:, None]))
    s['target'], s['peak'], credit = settlement_step(s['target'], s['peak'], expected,
        alignment, bottom-plane, velocity, force, active, reset,
        airborne_weight=airborne_weight, force_threshold=force_threshold)
    s.update(credit=credit.sum(1), step=int(env.common_step_counter), length=env.episode_length_buf.clone())
    s['total'] += s['credit']
    s['pending'][:] = False
    return s['credit']/float(env.step_dt)


def landing_reward(env):
    """V1-compatible landing credit with the historical descent mix."""
    return _landing_reward(env)


def landing_reward_load_biased(env):
    """V2 credit: descent starts learning, load-bearing contact finishes it."""
    return _landing_reward(env, airborne_weight=.15, force_threshold=15.)


def ascent_speed_floor(env, target_speed_fraction=.8):
    """Positive forward-speed opportunity reward on ascending flights only.

    The native world-route terms remain unchanged. This bounded additive term
    gives the fully trainable V2 actor a direct signal to keep advancing at
    >=80% of its command while it learns the missing per-step support timing.
    It has no action-amplitude or stand-still incentive.
    """
    from .v111_refine import state
    from .v135_terrain_geometry import v135_terrain_geometry

    state(env)
    entry = env._v111_first_tread
    geometry = v135_terrain_geometry(env)
    robot = env.scene['robot']
    path_speed = (robot.data.root_lin_vel_w[:, :2]*entry['forward']).sum(1)
    root_x = ((robot.data.root_pos_w[:, :2]-entry['anchor'])*entry['forward']).sum(1)
    # B1 is allowed to decelerate. After entry, an earlier skipped tread must
    # not erase the incentive to keep moving and recover later support.
    active = (geometry['ascent'] & entry['height_valid'] & ~entry['entry_near']
        & (root_x >= 0) & (root_x < geometry['flight_riser_count']*entry['tread']))
    command = env.command_manager.get_command('base_velocity')
    active &= (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03)
    return ascent_speed_credit(path_speed, command[:, 0], active, target_speed_fraction)


def reset_targeted(env, env_ids):
    from .v111_refine_commands import reset_refinement
    reset_refinement(env, env_ids)
    if hasattr(env, '_v111_targeted'):
        env._v111_targeted['pending'][env_ids] = True


def credit_metric(env, env_ids):
    landing_reward(env)
    return env._v111_targeted['total'][env_ids].mean()


def credit_metric_load_biased(env, env_ids):
    landing_reward_load_biased(env)
    return env._v111_targeted['total'][env_ids].mean()
