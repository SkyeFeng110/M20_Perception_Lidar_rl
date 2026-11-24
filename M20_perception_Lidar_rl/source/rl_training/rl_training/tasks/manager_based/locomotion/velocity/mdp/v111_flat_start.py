"""Flat/entry repair stage layered over the retained entry-transfer objective."""
import os
import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from rl_training.flat_start import direct_entry_stress_mask, filter_forward, flat_start_terms
from .v111_traction_transition import TractionTransitionVelocityCommand


class StoppedWheelVelocityAction(JointVelocityAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        command = self._env.command_manager.get_command('base_velocity')
        self._env._m20_last_action_command = command.clone()
        stop = torch.linalg.vector_norm(command, dim=1) < .08
        self._processed_actions[stop] = 0


def executed_last_action(env):
    # Same postprocessing/history convention as the robot; PPO still stores
    # its original sampled action/log-probability, so its likelihood is valid.
    action = env.action_manager.action.clone()
    command = getattr(env, '_m20_last_action_command', None)
    if command is not None:
        action[torch.linalg.vector_norm(command, dim=1) < .08, 12:] = 0
    return action


class FlatStartVelocityCommand(TractionTransitionVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.filtered_forward = torch.zeros(self.num_envs, device=self.device)
        self.requested_forward = self.filtered_forward.clone()
        self._forward_target = self.filtered_forward.clone()
        from .v135_terrain_geometry import column_names, ascent_name
        names = column_names(env.cfg.scene.terrain.terrain_generator)
        self._flat_columns = torch.tensor([n == 'flat' for n in names], device=self.device)
        self._ascent_columns = torch.tensor([ascent_name(n) for n in names], device=self.device)
        self._direct_entry_repair = os.environ.get('M20_ENTRY_DIRECT_REPAIR') == '1'

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._forward_target[ids] = self.vel_command_b[ids, 0]
        # A command resample while walking must not reset a moving filter.
        new_episode = ids[self._env.episode_length_buf[ids] == 0]
        self.filtered_forward[new_episode] = 0

    def _update_command(self):
        # Parents retain the native request in vel_command_b between updates.
        self.vel_command_b[:, 0] = self._forward_target
        super()._update_command()
        self._forward_target.copy_(self.vel_command_b[:, 0])
        self.requested_forward.copy_(self._forward_target)
        self.filtered_forward.copy_(filter_forward(self.filtered_forward,
            self.requested_forward, self._env.step_dt))
        # Stress training only: one of the four flat/ascent transition cohorts
        # keeps the current robot's raw request. Candidate deployment filters.
        stress = direct_entry_stress_mask(
            torch.arange(self.num_envs, device=self.device),
            self._env.scene.terrain.terrain_types,
            self._flat_columns, self._ascent_columns,
            enabled=self._direct_entry_repair)
        self.vel_command_b[:, 0] = torch.where(stress, self.requested_forward, self.filtered_forward)


def reset(env, env_ids):
    from .v111_entry_transfer import reset as entry_reset
    entry_reset(env, env_ids)
    if hasattr(env, '_m20_flat_start'):
        env._m20_flat_start['pending'][env_ids] = True


def state(env):
    from .v111_traction_transition import state as traction_state
    from .v135_terrain_geometry import column_names
    traction_state(env)
    step = int(env.common_step_counter)
    robot = env.scene['robot']
    if not hasattr(env, '_m20_flat_start'):
        zeros = lambda: torch.zeros(env.num_envs, device=env.device)
        # Static true flat columns only: no stair-approach/landing classification
        # from noisy/absent perception. This cannot affect B1 or B2+ leg costs.
        names = column_names(env.cfg.scene.terrain.terrain_generator)
        columns = torch.tensor([n == 'flat' for n in names], device=env.device)
        hip_ids = [robot.joint_names.index(leg+'_hipx_joint') for leg in ('fl', 'fr', 'hl', 'hr')]
        env._m20_flat_start = dict(step=-1, window=zeros(), previous_command=zeros(),
            last=env.episode_length_buf.clone(), pending=torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
            columns=columns, hip_ids=hip_ids)
    s = env._m20_flat_start
    if s['step'] == step:
        return s
    command = env.command_manager.get_command('base_velocity')
    term = env.command_manager.get_term('base_velocity')
    active = (s['columns'][env.scene.terrain.terrain_types]
              & (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03)
              & ~term.is_heading_env)
    wheel = robot.data.body_link_pos_w[:, env._v111_recovery_wheels]
    relative = wheel-robot.data.root_pos_w[:, None]
    wheel_b = quat_apply_inverse(robot.data.root_quat_w[:, None].expand(-1, 4, -1), relative)
    gravity = robot.data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=1))
    reset_mask = s['pending'] | (env.episode_length_buf < s['last']) | (env.episode_length_buf == 0)
    values = flat_start_terms(wheel_b[..., 1], robot.data.joint_pos[:, s['hip_ids']],
        pitch, robot.data.root_ang_vel_b[:, 1], env.action_manager.action,
        env.action_manager.prev_action, command[:, 0], s['previous_command'],
        s['window'], active, reset_mask, float(env.step_dt))
    s.update(values, step=step, last=env.episode_length_buf.clone(),
             previous_command=command[:, 0].detach().clone())
    s['pending'][:] = False
    return s


def safety_cost(env):
    return -state(env)['cost']


def metric(env, env_ids, key):
    s = state(env)
    mask = s['active'][env_ids]
    return (s[key][env_ids].float()*mask).sum()/mask.sum().clamp_min(1)
