"""Retain model 47900's support while learning steering and missing treads."""
import os
from pathlib import Path
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from .v111_step_support_env_cfg import (
    DeeproboticsM20V111StepSupportEnvCfg, DeeproboticsM20V111StepSupportPPORunnerCfg,
    M20StepSupportPPOCfg)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_step_retention import route_reward


@configclass
class M20StepRetentionPPOCfg(M20StepSupportPPOCfg):
    class_name: str = 'rl_training.v111_step_retention_runner:StepRetentionPPO'
    behavior_reference_results: str = ''
    success_anchor_weight: float = 2.
    local_anchor_weight: float = 1.
    num_learning_epochs: int = 2
    learning_rate: float = 5.e-6


@configclass
class DeeproboticsM20V111StepRetentionEnvCfg(DeeproboticsM20V111StepSupportEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Native forward reward and post-entry speed target stay live even
        # after a skip. Standstill cannot earn this additional positive term.
        self.rewards.v111_retention_forward_route = RewTerm(func=route_reward, weight=.75)


@configclass
class DeeproboticsM20V111StepRetentionPPORunnerCfg(DeeproboticsM20V111StepSupportPPORunnerCfg):
    experiment_name = 'M20-V11.1-StepRetention'
    algorithm = M20StepRetentionPPOCfg()

    def __post_init__(self):
        super().__post_init__()
        # Ancestor __post_init__ overwrites learning rate for old profiles.
        self.algorithm.learning_rate = 5.e-6
        path = os.environ.get('M20_RETENTION_RESULTS', '')
        if not path or not Path(path).is_file():
            raise ValueError('StepRetention requires M20_RETENTION_RESULTS from its teacher benchmark')
        self.algorithm.behavior_reference_results = str(Path(path).resolve())
