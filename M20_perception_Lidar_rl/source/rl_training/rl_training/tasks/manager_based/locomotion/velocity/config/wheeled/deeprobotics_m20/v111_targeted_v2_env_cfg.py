"""V2 targeted correction: full actor relearning with speed/contact safeguards."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .v111_targeted_env_cfg import DeeproboticsM20V111TargetedEnvCfg, M20TargetedPPOCfg
from .v111_refine_env_cfg import DeeproboticsM20V111RefinePPORunnerCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_targeted as targeted


@configclass
class M20TargetedV2PPOCfg(M20TargetedPPOCfg):
    class_name: str = 'rl_training.v111_targeted_v2_runner:TargetedPPOV2'
    std_floor_leg: float = .18
    std_floor_wheel: float = .35


@configclass
class DeeproboticsM20V111TargetedV2EnvCfg(DeeproboticsM20V111TargetedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Reward-side contact credit is mostly reserved for >=15N loading.
        self.rewards.v111_refine_landing.func = targeted.landing_reward_load_biased
        self.curriculum.v111_refine_support_credit_sum = CurrTerm(func=targeted.credit_metric_load_biased)
        # Positive bounded speed floor preserves forward opportunity without a
        # global slow-down penalty or action-amplitude cost.
        self.rewards.v111_targeted_ascent_speed_floor = RewTerm(
            func=targeted.ascent_speed_floor, weight=.75)


@configclass
class DeeproboticsM20V111TargetedV2PPORunnerCfg(DeeproboticsM20V111RefinePPORunnerCfg):
    experiment_name = 'M20-V11.1-TargetedCorrectionV2'
    algorithm = M20TargetedV2PPOCfg()
