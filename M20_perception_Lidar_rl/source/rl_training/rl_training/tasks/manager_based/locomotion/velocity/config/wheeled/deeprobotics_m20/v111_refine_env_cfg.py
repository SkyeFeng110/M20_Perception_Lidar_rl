"""Separate recovery refinement profile; historical AscentRecovery is reproducible."""
import math
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm
from .v111_recovery_env_cfg import DeeproboticsM20V111RecoveryEnvCfg, DeeproboticsM20V111RecoveryPPORunnerCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_refine as refine
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_refine_commands import RefineVelocityCommand, reset_refinement


@configclass
class M20V111RefineCfg:
    route_yaw_scale: float = .10
    route_lateral_scale: float = .20
    route_weight: float = .20
    landing_weight: float = .20
    arrival_weight: float = .50
    contact_weight: float = .15


@configclass
class DeeproboticsM20V111RefineEnvCfg(DeeproboticsM20V111RecoveryEnvCfg):
    v111_refine: M20V111RefineCfg = M20V111RefineCfg()

    def __post_init__(self):
        super().__post_init__()
        c = self.v111_refine
        if any(not math.isfinite(v) or v <= 0 for v in vars(c).values() if isinstance(v, (float, int))):
            raise ValueError('Refinement scales must be positive and finite')
        # NEW contact cost cannot exceed .15 reward/second. Chatter is trained
        # on policy means in PPO, not charged to fixed Gaussian exploration.
        if c.contact_weight > .15 or c.route_weight > .20:
            raise ValueError('Refinement cost budget exceeded')
        self.commands.base_velocity.class_type = RefineVelocityCommand
        self.events.v111_refine_reset = EventTerm(func=reset_refinement, mode='reset')
        self.rewards.v111_recovery_flat_quality.func = refine.quality_reward
        self.rewards.v111_refine_route = RewTerm(func=refine.route_cost, weight=c.route_weight)
        self.rewards.v111_refine_landing = RewTerm(func=refine.landing_reward, weight=c.landing_weight)
        self.rewards.v111_refine_arrival = RewTerm(func=refine.arrival_reward, weight=c.arrival_weight)
        self.rewards.v111_quiet_contact = RewTerm(func=refine.contact_cost, weight=c.contact_weight)
        for key in ('noise_ramp', 'contact_sum', 'chatter_sum', 'touch_sum', 'support_credit_sum',
                    'route_sum', 'yaw_max', 'reach_max', 'arrived'):
            setattr(self.curriculum, 'v111_refine_'+key, CurrTerm(func=refine.metric, params={'key': key}))


@configclass
class DeeproboticsM20V111RefinePPORunnerCfg(DeeproboticsM20V111RecoveryPPORunnerCfg):
    experiment_name = 'M20-V11.1-RecoveryRefine'

    def __post_init__(self):
        super().__post_init__()
        # Physical ascent labels, including the idle/approach phase, release
        # defective leg and wheel-differential outputs. Common drive is kept.
        self.algorithm.anchor_physical_ascent = True
        self.algorithm.anchor_ascent_wheel_differential_weight = .08
        self.algorithm.quiet_mean_weight = .02
