"""Focus on full-speed flat posture while retaining the current stair gait."""
import os
from pathlib import Path
from rl_training.joint_target_limits import joint_target_contract
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from .v111_entry_transfer_env_cfg import (DeeproboticsM20V111EntrySequenceEnvCfg,
    DeeproboticsM20V111EntrySequencePPORunnerCfg, M20EntrySequencePPOCfg)
from rl_training.tasks.manager_based.locomotion.velocity.mdp import (
    v111_flat_posture as flat,
    v111_joint_limits as joint_limits,
)


@configclass
class DeeproboticsM20V111FlatPostureEnvCfg(DeeproboticsM20V111EntrySequenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Pause further B1-sequence optimization for this phase. The currently
        # accepted entry/later gait is retained with measured moving replay.
        for name in ('v111_entry_sequence', 'v111_entry_transfer_progress',
                     'v111_entry_transfer_safety', 'v111_sequence_flat_motion_cost',
                     'v111_sequence_flat_motion_quality'):
            setattr(self.rewards, name, None)
        for name in tuple(vars(self.curriculum)):
            if name.startswith('v111_sequence_flat_'):
                setattr(self.curriculum, name, None)
        self.rewards.v111_flat_posture_cost = RewTerm(func=flat.cost, weight=1.)
        self.rewards.v111_flat_posture_quality = RewTerm(func=flat.quality, weight=1.)
        for key in ('reach', 'pitch', 'pitch_rate', 'speed', 'yaw', 'lateral',
                    'reach_peak', 'pitch_peak', 'pitch_rate_peak', 'onset_event_cost',
                    'active_samples', 'eligible_episodes', 'reach_sum', 'pitch_sum',
                    'pitch_rate_sum', 'speed_sum', 'yaw_sum', 'lateral_sum'):
            setattr(self.curriculum, 'v111_flat_posture_'+key,
                    CurrTerm(func=flat.metric, params={'key': key}))
        target_contract = joint_target_contract()
        self.actions.joint_pos.class_type = joint_limits.LimitedLegPositionAction
        self.actions.joint_pos.clip = joint_limits.joint_target_clip()
        self.observations.policy.actions.func = joint_limits.executed_last_action
        self.observations.critic.actions.func = joint_limits.executed_last_action
        self.rewards.v111_joint_target_limit = RewTerm(
            func=joint_limits.joint_target_excess_cost,
            weight=target_contract['excess_cost_weight'])
        self.curriculum.v111_joint_target_violation_fraction = CurrTerm(
            func=joint_limits.joint_target_violation_fraction)
        self.commands.base_velocity.class_type = flat.FlatPostureVelocityCommand
        self.events.v111_refine_reset.func = flat.reset
        # Keep the original ascent share; add flat practice by reducing
        # other terrain shares. This phase is not a new staircase curriculum.
        terrain = self.scene.terrain.terrain_generator.sub_terrains
        terrain['flat'].proportion = .4
        for name in ('boxes', 'random_rough', 'hf_pyramid_slope', 'hf_pyramid_slope_inv'):
            terrain[name].proportion = .025


@configclass
class M20FlatPosturePPOCfg(M20EntrySequencePPOCfg):
    class_name: str = 'rl_training.v111_flat_posture_runner:FlatPosturePPO'
    compact_replay: str = ''
    compact_anchor_weight: float = .5
    local_anchor_weight: float = .5
    success_anchor_weight: float = 1.
    joint_target_contract: dict = None


@configclass
class DeeproboticsM20V111FlatPosturePPORunnerCfg(DeeproboticsM20V111EntrySequencePPORunnerCfg):
    experiment_name = 'M20-V11.1-FlatPosture'
    algorithm = M20FlatPosturePPOCfg()

    def __post_init__(self):
        super().__post_init__()
        path = Path(os.environ.get('M20_COMPACT_REPLAY', ''))
        if not path.is_file():
            raise ValueError('FlatPosture requires M20_COMPACT_REPLAY')
        self.algorithm.compact_replay = str(path.resolve())
        self.algorithm.joint_target_contract = joint_target_contract()
        self.algorithm.learning_rate = 1.e-5
