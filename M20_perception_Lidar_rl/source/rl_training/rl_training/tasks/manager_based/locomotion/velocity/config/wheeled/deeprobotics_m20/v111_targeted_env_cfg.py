"""Targeted correction profile for the next manually started stage."""
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from .v111_refine_env_cfg import DeeproboticsM20V111RefineEnvCfg, DeeproboticsM20V111RefinePPORunnerCfg
from .v111_recovery_env_cfg import M20RecoveryPPOCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_targeted as targeted


@configclass
class M20TargetedPPOCfg(M20RecoveryPPOCfg):
    class_name: str = 'rl_training.v111_targeted_runner:TargetedPPO'
    symmetry_weight: float = .05
    idle_mean_weight: float = .10
    flat_mean_weight: float = .03


@configclass
class DeeproboticsM20V111TargetedEnvCfg(DeeproboticsM20V111RefineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.v111_adjacent.support_weight = 1.60
        self.v111_refine.landing_weight = .30
        self.v111_entry_flat.entry_support_weight = 2.50
        self.rewards.v111_entry_support.weight = self.v111_entry_flat.entry_support_weight
        self.rewards.v111_adjacent_support.weight = self.v111_adjacent.support_weight
        self.rewards.v111_refine_landing.weight = self.v111_refine.landing_weight
        self.rewards.v111_refine_landing.func = targeted.landing_reward
        self.events.v111_refine_reset.func = targeted.reset_targeted
        self.curriculum.v111_refine_support_credit_sum = CurrTerm(func=targeted.credit_metric)


@configclass
class DeeproboticsM20V111TargetedPPORunnerCfg(DeeproboticsM20V111RefinePPORunnerCfg):
    experiment_name = 'M20-V11.1-TargetedCorrection'
    algorithm = M20TargetedPPOCfg()
