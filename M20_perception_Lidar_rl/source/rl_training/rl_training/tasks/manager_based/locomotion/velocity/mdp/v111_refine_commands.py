"""Standing policy takeover -> idle -> forward samples, with native 3-D commands.

This covers the C-to-8 transition. The external, prescribed prone-to-Z pose
controller is still tested in MuJoCo, and is not injected into PPO actions.
"""
import torch
from .commands import UniformThresholdVelocityCommand


def takeover_fraction(elapsed, idle=2., ramp=.6):
    return ((elapsed-idle)/ramp).clamp(0, 1)


class RefineVelocityCommand(UniformThresholdVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.refine_cohort = torch.arange(self.num_envs, device=self.device) % 4 == 0
        self.refine_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.refine_speed = torch.zeros_like(self.refine_elapsed)
        self.refine_heading = torch.zeros_like(self.refine_elapsed)
        self.refine_origin = torch.zeros(self.num_envs, 2, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.refine_heading[ids] = self.robot.data.heading_w[ids]
        self.refine_origin[ids] = self.robot.data.root_pos_w[ids, :2]
        self.refine_elapsed[ids] = 0
        cohort = ids[self.refine_cohort[ids]]
        self.refine_speed[cohort] = torch.empty(len(cohort), device=self.device).uniform_(.5, 1.)
        self.vel_command_b[cohort] = 0
        self.is_heading_env[cohort] = False
        self.is_standing_env[cohort] = False

    def _update_command(self):
        super()._update_command()
        ids = self.refine_cohort
        fraction = takeover_fraction(self.refine_elapsed[ids])
        self.vel_command_b[ids, 0] = self.refine_speed[ids]*fraction
        self.vel_command_b[ids, 1:] = 0
        self.refine_elapsed += float(self._env.step_dt)


def reset_refinement(env, env_ids):
    """Nominal stationary takeover on 25%; keep randomized resets on 75%."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if hasattr(env, '_v111_refine'):
        env._v111_refine['pending'][ids] = True
    cohort = ids[ids % 4 == 0]
    if not len(cohort):
        return
    robot = env.scene['robot']
    root = robot.data.root_state_w[cohort].clone()
    yaw = robot.data.heading_w[cohort]
    root[:, 3:7] = 0
    root[:, 3] = (yaw/2).cos()
    root[:, 6] = (yaw/2).sin()
    root[:, 7:13] = 0
    robot.write_root_state_to_sim(root, env_ids=cohort)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[cohort],
                                  torch.zeros_like(robot.data.default_joint_vel[cohort]), env_ids=cohort)
