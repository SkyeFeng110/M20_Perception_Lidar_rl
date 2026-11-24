"""Sequential support stage: denser load credit and matched ascent practice."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from .v111_targeted_v2_env_cfg import (
    DeeproboticsM20V111TargetedV2EnvCfg, DeeproboticsM20V111TargetedV2PPORunnerCfg,
    M20TargetedV2PPOCfg)
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_step_support as support
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import ascent_name


@configclass
class M20StepSupportPPOCfg(M20TargetedV2PPOCfg):
    # Credit from lift to sustained support spans several control ticks.
    lam: float = .97


@configclass
class DeeproboticsM20V111StepSupportEnvCfg(DeeproboticsM20V111TargetedV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Six ascent columns retain their total 30% share. Four practice the
        # deployment benchmark, two retain the original height curriculum.
        widths = (.26, .30, .26, .30, .24, .34)
        self.v111_entry_flat.ascent_treads = widths
        variants = [t for name, t in self.scene.terrain.terrain_generator.sub_terrains.items()
                    if ascent_name(name)]
        if len(variants) != len(widths):
            raise ValueError('StepSupport expects the six verified ascent columns')
        for index, (terrain, width) in enumerate(zip(variants, widths)):
            terrain.step_width = width
            if index < 4:
                terrain.step_height_range = (.16, .16)
        self.commands.base_velocity.class_type = support.StepSupportVelocityCommand
        self.rewards.v111_refine_landing.func = support.landing_reward
        self.v111_refine.landing_weight = .60
        self.rewards.v111_refine_landing.weight = self.v111_refine.landing_weight
        self.v111_adjacent.support_weight = 2.40
        self.rewards.v111_adjacent_support.weight = self.v111_adjacent.support_weight
        # Preserve all native velocity/progress terms and finite failure
        # budgets. This stage adds positive support opportunity, no new cost.
        self.events.v111_refine_reset.func = support.reset
        self.curriculum.v111_refine_support_credit_sum = CurrTerm(
            func=support.metric, params={'key': 'credit_sum'})
        for key in ('eligible_episodes', 'supported_fraction', 'complete_fraction',
                    'b1_both_fraction', 'missed'):
            setattr(self.curriculum, 'v111_step_'+key,
                    CurrTerm(func=support.metric, params={'key': key}))


@configclass
class DeeproboticsM20V111StepSupportPPORunnerCfg(DeeproboticsM20V111TargetedV2PPORunnerCfg):
    experiment_name = 'M20-V11.1-StepSupport'
    algorithm = M20StepSupportPPOCfg()
