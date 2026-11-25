"""Privileged reward masks only; the actor retains the native 63 observations."""
import torch
from isaaclab.utils.math import quat_apply_inverse
from rl_training.flat_posture import profiles, posture_terms, onset_step
from .v111_entry_transfer import EntrySequenceVelocityCommand


class FlatPostureVelocityCommand(EntrySequenceVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        focused, _, _ = profiles(ids, self.refine_elapsed[ids])
        chosen = ids[focused & self._flat_sequence_columns[self._env.scene.terrain.terrain_types[ids]]]
        self.vel_command_b[chosen] = 0
        self.is_heading_env[chosen] = False
        self.is_standing_env[chosen] = False

    def _update_command(self):
        super()._update_command()
        ids = torch.arange(self.num_envs, device=self.device)
        focused, _, speed = profiles(ids, self.refine_elapsed)
        mask = focused & self._flat_sequence_columns[self._env.scene.terrain.terrain_types]
        self.vel_command_b[mask, 0] = speed[mask]
        self.vel_command_b[mask, 1:] = 0


def reset(env, env_ids):
    from .v111_entry_transfer import reset as parent_reset
    from .v135_terrain_geometry import column_names
    parent_reset(env, env_ids)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    columns = torch.tensor([n == 'flat' for n in column_names(
        env.cfg.scene.terrain.terrain_generator)], device=env.device)
    _, nominal, _ = profiles(ids, torch.zeros(len(ids), device=env.device))
    selected = ids[nominal & columns[env.scene.terrain.terrain_types[ids]]]
    if len(selected):
        robot = env.scene['robot']
        root = robot.data.root_state_w[selected].clone()
        yaw = robot.data.heading_w[selected]
        root[:, 3:7] = 0
        root[:, 3], root[:, 6] = (yaw/2).cos(), (yaw/2).sin()
        root[:, 7:13] = 0
        robot.write_root_state_to_sim(root, env_ids=selected)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[selected],
            torch.zeros_like(robot.data.default_joint_vel[selected]), env_ids=selected)
    if hasattr(env, '_m20_flat_posture'):
        env._m20_flat_posture['pending'][ids] = True


def state(env):
    from .v111_entry_transfer import state as entry_state
    from .v135_terrain_geometry import column_names, v135_terrain_geometry
    entry_state(env)
    robot, term = env.scene['robot'], env.command_manager.get_term('base_velocity')
    if not hasattr(env, '_m20_flat_posture'):
        n, d = env.num_envs, env.device
        env._m20_flat_posture = dict(step=-1,
            columns=torch.tensor([name == 'flat' for name in column_names(
                env.cfg.scene.terrain.terrain_generator)], device=d),
            hips=[robot.find_bodies(leg+'_hipy')[0][0] for leg in ('fl', 'fr', 'hl', 'hr')],
            window=torch.zeros(n, device=d), peaks=torch.zeros(n, 4, device=d),
            previous_command=torch.zeros(n, device=d),
            pending=torch.ones(n, device=d, dtype=torch.bool),
            length=env.episode_length_buf.clone(),
            samples=torch.zeros(n, device=d), sums=torch.zeros(n, 6, device=d),
            maxima=torch.zeros(n, 3, device=d), onset_sum=torch.zeros(n, device=d))
    s = env._m20_flat_posture
    if s['step'] == int(env.common_step_counter):
        return s
    reset_mask = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    wheels = robot.data.body_link_pos_w[:, env._v111_recovery_wheels]
    q = robot.data.root_quat_w[:, None].expand(-1, 4, -1)
    reach = quat_apply_inverse(q, wheels-robot.data.body_link_pos_w[:, s['hips']])[..., 0]
    wheel_y = quat_apply_inverse(q, wheels-robot.data.root_pos_w[:, None])[..., 1]
    # A far, level stair approach can share flat safety. Release it 0.5m
    # before the first riser: neither B1 transfer nor later gait is reshaped.
    entry = env._v111_first_tread
    forward_x = ((wheels[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    # The verified inverted-pyramid mesh origin is its bottom platform. Use
    # that physical plane even before the first riser enters scanner range.
    far_approach = (v135_terrain_geometry(env)['ascent'] & (forward_x[:, :2].amax(1) < -.50)
                    & ((wheels[..., 2]-env.cfg.v11_perception.wheel_radius
                        -env.scene.env_origins[:, None, 2]).abs().amax(1) < .12))
    straight = ((term.vel_command_b[:, 1].abs() < .05)
                & (term.vel_command_b[:, 2].abs() < .03) & ~term.is_heading_env)
    physical = s['columns'][env.scene.terrain.terrain_types] | far_approach
    active = physical & straight
    command = env.command_manager.get_command('base_velocity')
    yaw = robot.data.heading_w-term.refine_heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    offset = robot.data.root_pos_w[:, :2]-term.refine_origin
    lateral = -offset[:, 0]*term.refine_heading.sin()+offset[:, 1]*term.refine_heading.cos()
    speed = (robot.data.root_lin_vel_w[:, 0]*term.refine_heading.cos()
             +robot.data.root_lin_vel_w[:, 1]*term.refine_heading.sin())
    gravity = robot.data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=1))
    s.update(posture_terms(reach, yaw, lateral, speed, command[:, 0], pitch,
                          robot.data.root_ang_vel_b[:, 1], wheel_y, active))
    s['window'], s['peaks'], s['onset_cost'] = onset_step(s['window'], s['peaks'],
        s['previous_command'], command[:, 0], env.action_manager.action,
        env.action_manager.prev_action, robot.data.root_ang_vel_b[:, 1], wheel_y,
        active, reset_mask, float(env.step_dt))
    # Episode aggregates, not just the final reset frame. Empty eligibility
    # batches log NaN neither; counts let the dashboard expose sample scarcity.
    for key in ('samples', 'sums', 'maxima', 'onset_sum'):
        s[key][reset_mask] = 0
    valid = s['active'] & ~reset_mask
    s['samples'] += valid
    values = torch.stack([s[k] for k in ('reach', 'pitch', 'pitch_rate', 'speed', 'yaw', 'lateral')], 1)
    s['sums'] += values*valid[:, None]
    s['maxima'] = torch.maximum(s['maxima'], values[:, :3]*valid[:, None])
    s['onset_sum'] += s['onset_cost']*float(env.step_dt)
    s.update(step=int(env.common_step_counter), previous_command=command[:, 0].clone(),
             length=env.episode_length_buf.clone())
    s['pending'][:] = False
    return s


def cost(env):
    s = state(env)
    return -s['cost']-s['onset_cost']


def quality(env):
    return state(env)['quality']


def metric(env, env_ids, key):
    s = state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if key == 'active_samples':
        return s['samples'][ids].sum()
    if key == 'eligible_episodes':
        return (s['samples'][ids] > 0).sum().float()
    if key == 'onset_event_cost':
        return s['onset_sum'][ids].sum()/(s['samples'][ids] > 0).sum().clamp_min(1)
    keys = ('reach', 'pitch', 'pitch_rate', 'speed', 'yaw', 'lateral')
    if key.endswith('_sum'):
        return s['sums'][ids, keys.index(key[:-4])].sum()
    if key.endswith('_peak'):
        return s['maxima'][ids, keys.index(key[:-5])].max() if len(ids) else s['samples'].new_zeros(())
    return s['sums'][ids, keys.index(key)].sum()/s['samples'][ids].sum().clamp_min(1)
