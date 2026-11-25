"""Retain the acquired flat posture and stair gait while repairing route drift."""
import os
from pathlib import Path
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from .v111_flat_posture_env_cfg import (DeeproboticsM20V111FlatPostureEnvCfg,
    DeeproboticsM20V111FlatPosturePPORunnerCfg, M20FlatPosturePPOCfg)
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_route_hold as route


@configclass
class DeeproboticsM20V111RouteHoldEnvCfg(DeeproboticsM20V111FlatPostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Replace duplicate route terms on flat/ascent only. Other terrains
        # keep their native terms, and speed/stance/support rewards stay live.
        for name in ('path_lateral_deviation', 'path_heading_error', 'stair_straightness',
                     'v111_refine_route', 'v111_retention_forward_route', 'v111_straight_stability'):
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.params = {'reward_func': term.func, 'reward_params': dict(term.params)}
                term.func = route.outside_focus_reward
        self.rewards.v111_route_hold_cost = RewTerm(func=route.cost, weight=1.)
        self.rewards.v111_route_hold_quality = RewTerm(func=route.quality, weight=1.5)
        self.rewards.v111_route_early_traction_quality = RewTerm(
            func=route.early_traction_quality, weight=.60)
        self.rewards.v111_route_early_traction_cost = RewTerm(
            func=route.early_traction_cost, weight=.13)
        self.rewards.v111_flat_posture_cost.func = route.flat_posture_cost
        self.rewards.v111_flat_posture_quality.func = route.flat_posture_quality
        self.events.v111_refine_reset.func = route.reset
        for terrain in ('flat', 'ascent'):
            for key in ('samples', 'yaw', 'lateral', 'lateral_speed', 'yaw_rate', 'speed',
                        'yaw_peak', 'lateral_peak', 'lateral_speed_peak', 'yaw_rate_peak',
                        'yaw_sum', 'lateral_sum', 'lateral_speed_sum', 'yaw_rate_sum', 'speed_sum'):
                name = terrain+'_'+key
                setattr(self.curriculum, 'v111_route_hold_'+name,
                    CurrTerm(func=route.metric, params={'key': name}))
        for key in ('slip_mean', 'slip_samples', 'support_fraction',
                    'quality_mean', 'eligible_steps'):
            setattr(self.curriculum, 'v111_route_early_traction_'+key,
                CurrTerm(func=route.early_traction_metric, params={'key': key}))


@configclass
class M20RouteHoldPPOCfg(M20FlatPosturePPOCfg):
    class_name: str = 'rl_training.v111_route_hold_runner:RouteHoldPPO'
    route_flat_replay: str = ''
    traction_reference_results: str = ''
    route_steering_weight: float = .15


@configclass
class DeeproboticsM20V111RouteHoldPPORunnerCfg(DeeproboticsM20V111FlatPosturePPORunnerCfg):
    experiment_name = 'M20-V11.1-EarlyTraction'
    algorithm = M20RouteHoldPPOCfg()

    def __post_init__(self):
        super().__post_init__()
        path = Path(os.environ.get('M20_ROUTE_FLAT_REPLAY', ''))
        if not path.is_file():
            raise ValueError('RouteHold requires measured M20_ROUTE_FLAT_REPLAY')
        self.algorithm.route_flat_replay = str(path.resolve())
        traction_path = Path(os.environ.get('M20_ROUTE_TRACTION_RESULTS', ''))
        if not traction_path.is_file():
            raise ValueError('RouteHold requires measured M20_ROUTE_TRACTION_RESULTS')
        self.algorithm.traction_reference_results = str(traction_path.resolve())
        self.algorithm.learning_rate = 3.e-6
