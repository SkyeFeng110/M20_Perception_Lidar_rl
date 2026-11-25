"""M20 leg target bounds and executed-action history for the V11.1 policy.

Adapted from the colleague's root-level v111_flat_start.py. Existing flat-start
command and posture logic remains in its original MDP module.
"""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from rl_training.joint_target_limits import LEG_JOINT_NAMES, joint_target_bounds, joint_target_clip


class LimitedLegPositionAction(JointPositionAction):
    """Clip scaled, offset leg targets while preserving PPO's sampled action."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if tuple(self._joint_names) != LEG_JOINT_NAMES or self.action_dim != 12:
            raise ValueError("M20 leg targets require the named FL/FR/HL/HR action order")
        if self.cfg.clip is None or self._clip.shape[1:] != (12, 2):
            raise ValueError("M20 leg targets require an explicit per-joint position bound")
        expected = torch.tensor(joint_target_bounds(), device=self.device, dtype=self._clip.dtype)
        if not torch.allclose(self._clip, expected.unsqueeze(0).expand_as(self._clip)):
            raise ValueError("Training joint target bounds differ from the confirmed M20 safety ledgers")
        self.executed_actions = torch.zeros_like(self.raw_actions)
        self.target_excess_rad = torch.zeros_like(self.raw_actions)
        self.violation_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frames = torch.zeros_like(self.violation_frames)

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite M20 leg position action")
        super().process_actions(actions)
        requested = self.raw_actions * self._scale + self._offset
        self.target_excess_rad.copy_((requested - self.processed_actions).abs())
        self.executed_actions.copy_(torch.where(
            self.target_excess_rad > 0,
            (self.processed_actions - self._offset) / self._scale,
            self.raw_actions,
        ))
        self.frames += 1
        self.violation_frames += (self.target_excess_rad.amax(dim=1) > 0).long()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.executed_actions[ids] = 0
        self.target_excess_rad[ids] = 0
        self.violation_frames[ids] = 0
        self.frames[ids] = 0


def joint_target_excess_cost(env):
    excess = env.action_manager.get_term("joint_pos").target_excess_rad.amax(dim=1)
    return (excess / 0.10).clamp(max=5.0)


def joint_target_violation_fraction(env, env_ids):
    term = env.action_manager.get_term("joint_pos")
    return (term.violation_frames[env_ids].sum()
            / term.frames[env_ids].sum().clamp_min(1))


def executed_last_action(env):
    """Expose the clipped action to the actor and critic, matching execution."""
    action = env.action_manager.action.clone()
    position_term = env.action_manager.get_term("joint_pos")
    if isinstance(position_term, LimitedLegPositionAction):
        action[:, :position_term.action_dim] = position_term.executed_actions
    command = getattr(env, "_m20_last_action_command", None)
    if command is not None:
        action[torch.linalg.vector_norm(command, dim=1) < 0.08, 12:] = 0
    return action
