"""Independent next stage; historical StepRetention behavior remains intact."""
import os
from pathlib import Path
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from .v111_step_retention_env_cfg import (
    DeeproboticsM20V111StepRetentionEnvCfg, DeeproboticsM20V111StepRetentionPPORunnerCfg,
    M20StepRetentionPPOCfg)
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_traction_transition as traction


@configclass
class M20TractionTransitionPPOCfg(M20StepRetentionPPOCfg):
    class_name: str = 'rl_training.v111_traction_transition_runner:TractionTransitionPPO'
    traction_reference_results: str = ''


@configclass
class DeeproboticsM20V111TractionTransitionEnvCfg(DeeproboticsM20V111StepRetentionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = traction.TractionTransitionVelocityCommand
        self.rewards.v111_traction_quality = RewTerm(func=traction.traction_quality, weight=.45)
        self.rewards.v111_traction_cost = RewTerm(func=traction.traction_cost, weight=.10)
        self.rewards.v111_transition_pitch = RewTerm(func=traction.pitch_cost, weight=.15)
        self.rewards.v111_transition_flat_quality = RewTerm(func=traction.flat_quality, weight=.35)
        self.rewards.v111_transition_action_impulse = RewTerm(func=traction.transition_impulse_penalty, weight=.35)
        self.rewards.v111_entry_pitch = RewTerm(func=traction.entry_pitch_penalty, weight=.45)
        self.rewards.v111_entry_simultaneous = RewTerm(func=traction.entry_simultaneous_penalty, weight=.80)
        self.rewards.v111_entry_lead = RewTerm(func=traction.entry_lead_reward, weight=.10)
        self.rewards.v111_retention_forward_route.weight = 1.
        for key in ('partial_rear_slip', 'full_rear_slip', 'horizontal_support_fraction', 'pitch_cost',
                    'transition_impulse', 'entry_pitch', 'entry_simultaneous', 'entry_lead'):
            setattr(self.curriculum, 'v111_traction_'+key, CurrTerm(func=traction.metric, params={'key': key}))


@configclass
class DeeproboticsM20V111TractionTransitionPPORunnerCfg(DeeproboticsM20V111StepRetentionPPORunnerCfg):
    experiment_name = 'M20-V11.1-TractionTransition'
    algorithm = M20TractionTransitionPPOCfg()

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.e-6
        path = os.environ.get('M20_TRACTION_RESULTS', '')
        if not path or not Path(path).is_file():
            raise ValueError('TractionTransition requires a measured M20_TRACTION_RESULTS baseline')
        self.algorithm.traction_reference_results = str(Path(path).resolve())
