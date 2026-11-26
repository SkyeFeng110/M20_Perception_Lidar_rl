"""V2 modifications applied only by the explicit descent/low-mu launcher."""
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm, EventTermCfg, SceneEntityCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_descent_traction as repair
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_front_turn as turn
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_friction_cohorts import WheelFrictionCohorts
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import column_names


def configure(env_cfg):
    generator = env_cfg.scene.terrain.terrain_generator
    before = column_names(generator)
    items = {}
    for name, source in generator.sub_terrains.items():
        if name != 'pyramid_stairs':
            items[name] = source
            continue
        for i, width in enumerate((.26, .30, .26, .30)):
            terrain = source.copy()
            terrain.proportion = float(source.proportion)/4.
            terrain.step_width = width
            if i < 2:
                terrain.step_height_range = (.16, .16)
            items[name if i == 0 else 'v2_descent_tread_'+str(i)] = terrain
    generator.sub_terrains = items
    after = column_names(generator)
    if any(a != b and not (repair.descent_name(a) and repair.descent_name(b)) for a, b in zip(before, after)):
        raise ValueError('Descent tread split changed flat/ascent terrain allocation')
    env_cfg.commands.base_velocity.class_type = repair.DescentRepairVelocityCommand
    env_cfg.events.v111_refine_reset.func = repair.reset
    # Appended after the native material event. Its 20% general cohort and
    # every non-wheel shape retain the original randomization.
    env_cfg.events.v2_wheel_friction = EventTermCfg(func=WheelFrictionCohorts, mode='startup', params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names=['.*_wheel']),
        static_friction_range=(.45, .70), dynamic_friction_range=(.45, .70),
        restitution_range=(0., 0.), num_buckets=256, make_consistent=True))
    for name in ('path_lateral_deviation', 'path_heading_error', 'stair_straightness',
                 'v111_refine_route', 'v111_retention_forward_route', 'v111_straight_stability'):
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            term.func = repair.outside_route_focus
    env_cfg.rewards.v2_descent_route_cost = RewTerm(func=repair.route_cost, weight=1.2)
    env_cfg.rewards.v2_descent_route_quality = RewTerm(func=repair.route_quality, weight=1.)
    env_cfg.rewards.v111_route_hold_cost.func = repair.ascent_route_cost
    env_cfg.rewards.v111_route_early_traction_quality = RewTerm(func=repair.slip_quality, weight=.70)
    env_cfg.rewards.v111_route_early_traction_cost = RewTerm(func=repair.slip_cost, weight=.18)
    # Max low-mu load credit at standstill is .70*.30*1.25*1.4=.3675.
    # The sustained-stall budget must exceed it even at maximum feedback.
    env_cfg.rewards.v111_early_handoff_stall.weight = .45
    env_cfg.rewards.v2_front_turn_cost = RewTerm(func=turn.cost, weight=1.)
    env_cfg.rewards.v2_front_turn_quality = RewTerm(func=turn.quality, weight=1.5)
    # Per-event cost (MDP divides by dt), kept below native yaw tracking.
    env_cfg.rewards.v2_front_turn_touchdown = RewTerm(func=turn.impact, weight=.12)
    # These old objectives resist the requested inward lean/lower stance.
    # Replace them only on physical flat terrain during deliberate turning.
    for name in ('flat_orientation_l2', 'hipx_joint_pos_penalty', 'hipy_joint_pos_penalty',
                 'knee_joint_pos_penalty', 'joint_mirror', 'pure_yaw_xy_drift'):
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            term.params = dict(reward_func=term.func, reward_params=dict(term.params))
            term.func = turn.TurnAwareOriginal
    term = env_cfg.rewards.feet_slide_ang_z_cmd
    term.func = turn.rolling_slip
    for key in ('samples', 'yaw_error', 'back_speed', 'drift', 'contact_fraction', 'front_share',
                'rear_unled', 'impact', 'slip', 'back_speed_peak', 'rear_force_peak'):
        setattr(env_cfg.curriculum, 'v2_front_turn_'+key, CurrTerm(func=turn.metric, params={'key': key}))
    for key in ('samples', 'yaw', 'lateral', 'side_speed', 'yaw_rate', 'speed',
                'yaw_peak', 'lateral_peak', 'yaw_signed', 'lateral_signed', 'low_mu_slip', 'low_mu_samples'):
        setattr(env_cfg.curriculum, 'v2_descent_traction_'+key,
                CurrTerm(func=repair.metric, params={'key': key}))
