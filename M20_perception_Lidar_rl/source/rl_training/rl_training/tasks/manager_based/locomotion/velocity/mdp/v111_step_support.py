"""V3 reward-side sequential support, with explicit conditional diagnostics."""
import torch
from rl_training.step_support import support_credit_step
from .v111_refine_commands import RefineVelocityCommand


class StepSupportVelocityCommand(RefineVelocityCommand):
    """Practice benchmark speeds on the four focused ascent columns only."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        from .v135_terrain_geometry import v135_terrain_geometry
        geometry = v135_terrain_geometry(self._env)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        focus = geometry['ascent'] & ((geometry['step_width'] == .26)
                                      | (geometry['step_width'] == .30))
        ids = ids[focus[ids]]
        speeds = self.vel_command_b.new_tensor([.5, .7, 1.])[
            torch.randint(3, (len(ids),), device=self.device)]
        cohort = self.refine_cohort[ids]
        self.refine_speed[ids[cohort]] = speeds[cohort]
        self.vel_command_b[ids] = 0
        self.vel_command_b[ids[~cohort], 0] = speeds[~cohort]
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False


def state(env):
    from .v111_refine import state as refine_state
    from .v135_terrain_geometry import v135_terrain_geometry
    refine_state(env)
    if not hasattr(env, '_v111_step_support'):
        z = lambda *shape: torch.zeros((env.num_envs, *shape), device=env.device)
        env._v111_step_support = dict(step=-1, target=z(2).long(), peak=z(2),
            frames=z(2).long(), pending=z().bool(), length=env.episode_length_buf.clone(),
            credit_sum=z(), support_sum=z(), opportunities=z(), missed=z(), eligible=z().bool())
    s = env._v111_step_support
    if s['step'] == int(env.common_step_counter):
        return s
    reset = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    if s['step'] == -1:
        reset[:] = True
    for key in ('target', 'peak', 'frames', 'credit_sum', 'support_sum', 'opportunities', 'missed', 'eligible'):
        s[key][reset] = 0
    entry, adjacent = env._v111_first_tread, env._v111_adjacent
    geometry = v135_terrain_geometry(env)
    robot = env.scene['robot']
    wheels = robot.data.body_pos_w[:, env._v111_recovery_wheels[:2]]
    x = ((wheels[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    expected = adjacent['overshoot_target']
    left = torch.stack((-entry['forward'][:, 1], entry['forward'][:, 0]), 1)
    lateral = ((wheels[..., :2]-env.scene.env_origins[:, None, :2])*left[:, None]).sum(-1)
    width = ((left.abs()*geometry['radii']).sum(-1)[:, None]
             +(expected-1).clamp_min(0)*entry['tread'][:, None]
             -env.cfg.v11_perception.wheel_radius-.03)
    command = env.command_manager.get_command('base_velocity')
    physical = ((geometry['ascent'] & entry['height_valid'])[:, None]
        & entry['entry_approached'] & (expected >= 1)
        & (expected <= geometry['flight_riser_count'][:, None]))
    active = physical & (command[:, 0] > .1)[:, None]
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts[:2], 2]
    velocity = robot.data.body_link_lin_vel_w[:, env._v111_recovery_wheels[:2], 2]
    credit = support_credit_step(s, expected, x,
        wheels[..., 2]-env.cfg.v11_perception.wheel_radius, velocity, force,
        entry['lower'], entry['height'], entry['tread'], lateral.abs() <= width,
        active, reset, float(env.step_dt))
    s['credit'] = credit.sum(1)
    # Events already have hard 3-frame certification; count only approached
    # ascent episodes in the dashboard instead of diluting with flat terrain.
    s['eligible'] |= (geometry['ascent'] & entry['height_valid']
                      & entry['entry_approached'].any(1))
    valid = ~reset
    s['credit_sum'] += s['credit']
    s['support_sum'] += (adjacent['support_event'].sum(1)
        + entry['entry_event'].sum(1))*valid
    s['missed'] = adjacent['missed_count'].sum(1).float()
    s['opportunities'] = adjacent['last_index'].sum(1).float()
    s.update(step=int(env.common_step_counter), length=env.episode_length_buf.clone())
    s['pending'][:] = False
    return s


def landing_reward(env):
    return state(env)['credit']/float(env.step_dt)


def reset(env, env_ids):
    from .v111_targeted import reset_targeted
    reset_targeted(env, env_ids)
    if hasattr(env, '_v111_step_support'):
        env._v111_step_support['pending'][env_ids] = True


def metric(env, env_ids, key):
    s = state(env)
    eligible = s['eligible'][env_ids]
    if key == 'eligible_episodes':
        return eligible.sum().float()
    if key == 'supported_fraction':
        support = (s['opportunities']-s['missed'])[env_ids][eligible].sum()
        return support/s['opportunities'][env_ids][eligible].sum().clamp_min(1)
    if key == 'complete_fraction':
        value = env._v111_adjacent['complete'][env_ids].float()
    elif key == 'b1_both_fraction':
        value = env._v111_first_tread['entry_certified'][env_ids].all(1).float()
    else:
        value = s[key][env_ids].float()
    return (value*eligible).sum()/eligible.sum().clamp_min(1)
