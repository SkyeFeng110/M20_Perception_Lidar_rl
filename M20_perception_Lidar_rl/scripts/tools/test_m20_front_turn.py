"""CPU counterexamples for turn rewards and untouched stair reward selection."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


turn = load('_front_turn_math', SOURCE/'front_turn.py')


def inputs():
    return dict(command=torch.tensor([[0., 0., .2], [0., 0., -.2], [0., 0., .2], [0., 0., .2]]),
        velocity=torch.tensor([[0., 0., 0.], [0., 0., 0.], [-.2, 0., 0.], [0., 0., 0.]]),
        yaw_rate=torch.tensor([.2, -.2, .2, 0.]),
        gravity=torch.tensor([[0., .016, -1.], [0., -.016, -1.], [0., .016, -1.], [0., .016, -1.]]),
        clearance=torch.full((4,), .548), home_height=torch.full((4,), .56),
        force=torch.full((4, 4), 100.), height=torch.zeros(4, 4),
        relative_velocity=torch.zeros(4, 4, 3), reach=torch.zeros(4, 4),
        rear_unled=torch.zeros(4), front_moment=torch.ones(4), rear_moment=torch.zeros(4),
        active=torch.ones(4, dtype=torch.bool))


def test_left_right_equal_backward_cost_and_no_stationary_turn_reward():
    result = turn.turn_terms(**inputs())
    assert result['cost'][0] == result['cost'][1]
    assert result['quality'][0] == result['quality'][1] > 0
    assert result['cost'][2] < result['cost'][0]
    assert result['quality'][3] == 0
    p = inputs(); p['active'][:] = False
    outside = turn.turn_terms(**p)
    assert not outside['cost'].any() and not outside['quality'].any()


def test_small_turn_prefers_loaded_wheels_without_demanding_lift_for_large_turns():
    p = inputs(); original = turn.turn_terms(**p)
    p['force'][:, 2] = 0
    lifted = turn.turn_terms(**p)
    assert (lifted['cost'] < original['cost']).all()
    p = inputs(); p['command'][:, 2] *= 3; p['yaw_rate'] = p['command'][:, 2].clone()
    loaded = turn.turn_terms(**p)
    assert (loaded['quality'] > 0).all()  # large yaw never requires a footstep


def test_rear_waits_for_loaded_front_initiation_and_can_follow_afterwards():
    force = torch.tensor([[100., 100., 0., 100.]])
    height = torch.tensor([[0., 0., .03, 0.]])
    result = turn.front_follow_step(torch.zeros(1), torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool), force, height, torch.zeros(1),
        torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), .02)
    assert result['rear_unled'].item() == .5
    for _ in range(4):
        result = turn.front_follow_step(result['ready_time'], result['front_started'], result['rear_released'],
            force, height, torch.full((1,), 2.), torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool), .02)
    assert result['rear_unled'].item() == 0
    # Reversal must not reuse the previous direction's front initiation.
    result = turn.front_follow_step(result['ready_time'], result['front_started'], result['rear_released'],
        force, height, torch.zeros(1), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), .02)
    assert result['rear_unled'].item() == .5


def test_touchdown_uses_precontact_speed_even_after_rebound():
    force = torch.full((1, 4), 250.)
    previous = torch.zeros_like(force)
    landing = torch.ones(1, 4, dtype=torch.bool)
    hard = turn.touchdown_risk(force, previous, torch.full_like(force, .1), torch.full_like(force, -.8), landing)
    soft = turn.touchdown_risk(force, previous, torch.zeros_like(force), torch.full_like(force, -.05), landing)
    assert hard.item() > soft.item()
    assert turn.touchdown_risk(force, previous, -force, -force, ~landing).item() == 0


def test_profile_has_pure_turns_stop_and_both_directions_without_reversing_vx():
    t = torch.tensor([0., 1., 4., 5., 8.])
    vx, wz = turn.turn_profile(t, torch.ones(5)*.4, torch.ones(5)*.3)
    assert wz.tolist() == pytest.approx([0., .4, 0., -.4, 0.])
    assert vx.tolist() == pytest.approx([0., .3, 0., .3, 0.])


def test_turn_gate_uses_physical_flat_column_and_wrapper_keeps_joint_selection(monkeypatch):
    class Entity:
        def __init__(self, name): self.name = name; self.joint_ids = slice(None)
        def resolve(self, scene): self.joint_ids = [0, 3, 6, 9]
    class Term:
        def __init__(self, cfg, env): pass
    for name in ('isaaclab', 'isaaclab.utils', '_front_mdp', 'rl_training'):
        package = types.ModuleType(name); package.__path__ = [str(SOURCE)]
        monkeypatch.setitem(sys.modules, name, package)
    managers = types.ModuleType('isaaclab.managers'); managers.SceneEntityCfg = Entity; managers.ManagerTermBase = Term
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    math_module = types.ModuleType('isaaclab.utils.math'); math_module.quat_apply_inverse = lambda q, v: v
    monkeypatch.setitem(sys.modules, math_module.__name__, math_module)
    geometry = types.ModuleType('_front_mdp.v135_terrain_geometry'); geometry.column_names = lambda cfg: ('flat', 'stairs')
    monkeypatch.setitem(sys.modules, geometry.__name__, geometry)
    module = load('_front_mdp.v111_front_turn', SOURCE/'tasks/manager_based/locomotion/velocity/mdp/v111_front_turn.py')
    wheels = torch.tensor([[.3, .25, .09], [.3, -.25, .09], [-.3, .25, .09], [-.3, -.25, .09]])
    pos = torch.cat((wheels, wheels+torch.tensor([0., 0., .4])), 0).repeat(3, 1, 1)
    names = [l+'_wheel' for l in ('fl', 'fr', 'hl', 'hr')]+[l+'_hipy' for l in ('fl', 'fr', 'hl', 'hr')]
    data = NS(body_link_pos_w=pos, body_link_lin_vel_w=torch.zeros(3, 8, 3), body_link_ang_vel_w=torch.zeros(3, 8, 3),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1), root_pos_w=torch.tensor([[0., 0., .56]]).repeat(3, 1),
        root_lin_vel_w=torch.zeros(3, 3), root_lin_vel_b=torch.zeros(3, 3),
        root_ang_vel_w=torch.zeros(3, 3), root_ang_vel_b=torch.tensor([[0., 0., .2]]).repeat(3, 1),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(3, 1))
    find = lambda name: ([names.index(name)], [name])
    sensor_data = NS(net_forces_w=torch.tensor([[[0., 0., 100.]]*4]*3))
    sensor_data.net_forces_w_history = sensor_data.net_forces_w[:, None].repeat(1, 3, 1, 1)
    class Scene(dict): pass
    scene = Scene(robot=NS(data=data, find_bodies=find))
    scene.sensors = {'contact_forces':NS(data=sensor_data, find_bodies=find)}
    scene.terrain = NS(terrain_types=torch.tensor([0, 1, 0])); scene.env_origins = torch.zeros(3, 3)
    cmd = torch.tensor([[0., 0., .2], [0., 0., .2], [1., 0., 0.]])
    env = NS(scene=scene, num_envs=3, device='cpu', common_step_counter=0, step_dt=.02,
        episode_length_buf=torch.ones(3, dtype=torch.long), command_manager=NS(get_command=lambda name:cmd),
        cfg=NS(scene=NS(terrain=NS(terrain_generator=None)), v11_perception=NS(wheel_radius=.09)))
    module.state(env); env.common_step_counter += 1; env.episode_length_buf += 1
    result = module.state(env)
    assert result['active'].tolist() == [True, False, False]
    entity = Entity('robot')
    cfg = NS(params={'reward_params':{'asset_cfg':entity}})
    wrapper = module.TurnAwareOriginal(cfg, env)
    assert entity.joint_ids == [0, 3, 6, 9]
    original = lambda env, asset_cfg: torch.full((3,), float(len(asset_cfg.joint_ids)))
    assert wrapper(env, original, cfg.params['reward_params']).tolist() == [0., 4., 4.]


def test_selection_cannot_trade_stopped_turn_for_quiet_wheels():
    import front_turn_selection as selection
    import m20_front_turn_benchmark as benchmark
    row = dict(forward=0., yaw=.2, friction=1., fell=False, yaw_tracking_ratio=1., yaw_error_mean=.02,
        backward_distance_m=.03, backward_speed_peak_m_s=.05, rear_force_peak_n=200.,
        pure_xy_speed_mean=.02,
        touchdown_down_speed_peak_m_s=.05, rear_lift_fraction=0., rear_unled_fraction=0.,
        forward_error_mean=.02, pitch_peak_deg=2., roll_peak_deg=3., contact_fraction=1.,
        front_positive_moment_share=.65)
    quiet = copy.deepcopy(row); quiet.update(yaw_tracking_ratio=0., yaw_error_mean=.2, rear_force_peak_n=100.)
    assert any('yaw tracking' in r for r in selection.regressions([quiet], [row]))
    assert selection.targets([row], [row]) == []
    # Sparse real falls remain behavior rows instead of evaluator exceptions.
    a = {k:np.zeros((2, 4)) for k in ('force', 'height', 'moment', 'touchdown', 'force_peak', 'touch_speed')}
    a.update(time=np.array([0., .02]), velocity=np.zeros((2, 3)), yaw_rate=np.zeros(2), pitch=np.zeros(2),
             roll=np.zeros(2), rear_unled=np.zeros((2, 2)))
    measured = benchmark.summarize(a, 0., .2, 1., True)
    assert measured['fell'] and measured['samples'] == 2
