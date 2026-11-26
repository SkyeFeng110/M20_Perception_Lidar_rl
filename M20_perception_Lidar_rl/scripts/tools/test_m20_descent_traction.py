"""CPU counterexamples for V2; no Isaac environment, physics or PPO updates."""
import copy
import importlib.util
import json
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


math = load('_v2_descent_math', SOURCE/'descent_traction.py')


def test_descent_correction_is_symmetric_and_keeps_gait_room():
    r = math.descent_route_terms(torch.tensor([.1, -.1, .1, .1]),
        torch.tensor([.15, -.15, .15, .15]), torch.tensor([.1, -.1, -.1, .1]),
        torch.tensor([.2, -.2, -.2, .2]), torch.tensor([.6, .6, .6, 0.]),
        torch.full((4,), .7), torch.ones(4, dtype=torch.bool))
    assert r['cost'][0] == r['cost'][1]
    assert r['cost'][2] > r['cost'][0]  # correction is allowed
    assert r['cost'][3] < 0 and r['quality'][3] == 0


@pytest.mark.parametrize('slip', [0., .3, .7, 1., 2.])
def test_slip_tail_has_no_air_or_unloading_escape(slip):
    v = torch.full((3, 4), slip)
    support = torch.ones(3, 4, dtype=torch.bool)
    speed, command = torch.tensor([0., .3, 1.]), torch.ones(3)
    a, b, _ = math.slip_tail_terms(v, support, support, speed, command)
    if slip:
        assert (b > 0).all()
    for wheel in range(4):
        mask = support.clone(); mask[:, wheel] = False
        c, d, _ = math.slip_tail_terms(v, mask, support, speed, command)
        assert (.70*c-.18*d < .70*a-.18*b).all()
    c, d, _ = math.slip_tail_terms(v, ~support, support, speed, command)
    assert not c.any() and not d.any()


def test_high_slip_cases_remain_distinguishable():
    v = torch.tensor([.5, .7, 1., 1.4])[:, None].expand(-1, 4)
    mask = torch.ones_like(v, dtype=torch.bool)
    _, cost, _ = math.slip_tail_terms(v, mask, mask, torch.ones(4), torch.ones(4))
    assert (torch.diff(cost) > 0).all()
    assert .70*.30*1.25*1.4 < .45  # max load credit cannot pay for sustained stopping


def test_retention_keeps_sagittal_gait_but_releases_descent_steering():
    pred = torch.ones(3, 16, requires_grad=True)
    target = torch.zeros_like(pred, requires_grad=True)
    early = torch.tensor([[False]*4, [True]*4, [False]*4])
    math.retention_loss(pred, target, early, torch.tensor([0, 1, 2])).backward()
    assert (pred.grad[:, math.SAGITTAL] > 0).all()
    assert not pred.grad[1, 12:].any()
    assert not pred.grad[2, math.HIPX].any()
    assert (pred.grad[0, 12:] > 0).all() and target.grad is None
    # A pure wheel differential on descent is never copied back to teacher.
    turn = torch.zeros(1, 16); turn[:, 12:] = torch.tensor([1., -1., 1., -1.])
    assert math.retention_loss(turn, torch.zeros_like(turn), torch.zeros(1, 4, dtype=torch.bool), torch.tensor([2])) == 0


def paths():
    ref = ROOT/'artifacts/early_traction_58098_20261125/reference58098'
    data = ROOT/'data/m20_recovery/early_traction_58098_20261125'
    return ref, data


def test_feedback_accumulates_for_drift_and_skips_without_a_fall(monkeypatch):
    package = types.ModuleType('_v2_feedback'); package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load(package.__name__+'.v111_descent_traction_runner', SOURCE/'v111_descent_traction_runner.py')
    alg = object.__new__(module.DescentTractionPPO)
    alg.keep_multiplier = dict(flat=1., ascent=1., descent=1.)
    alg.clean_streak = dict(flat=0, ascent=0, descent=0, traction=0, route=0)
    alg.descent_reward_multiplier = alg.slip_reward_multiplier = 1.
    alg.turn_reward_multiplier = 1.
    alg.turn_clean_streak = 0
    alg.base_learning_rate = alg.learning_rate = 1.5e-6
    alg.route_steering_weight = alg.route_base_steering_weight = .15
    alg.optimizer = NS(param_groups=[{'lr':alg.learning_rate}])
    alg._reward_env = NS()
    for _ in range(5):
        result = alg.behavior_feedback(['descent/0.26/0.7: yaw_peak_deg, lateral_peak_m',
            "ascent/('standing', 'ascent', .26, 1.): new skipped_treads"])
    assert alg.keep_multiplier['descent'] > 2. and alg.keep_multiplier['ascent'] > 2.
    assert alg.keep_multiplier['flat'] == 1.
    assert alg.learning_rate == pytest.approx(.75e-6)
    assert alg._reward_env._m20_v2_descent_multiplier > 1.
    assert result['action'] == 'record_and_continue'
    before = alg.keep_multiplier['descent']; alg.behavior_feedback([])
    assert alg.keep_multiplier['descent'] == before  # no immediate cancellation


def test_all_low_mu_speeds_are_ranked_and_missing_contact_is_rejected():
    from descent_traction_selection import low_mu_error, low_mu_errors
    path = ROOT/'artifacts/route_traction_20261125/reference58098/traction_b1_b3/results.json'
    seed = json.loads(path.read_text())
    improved = copy.deepcopy(seed)
    for row in improved:
        if row['terrain'] == 'ascent' and row['friction'] == .6:
            row['early_ascent_slip_p95'] *= .7
    assert low_mu_error(improved, seed) == pytest.approx(.7)
    bad = copy.deepcopy(improved)
    row = next(r for r in bad if r['terrain'] == 'ascent' and r['friction'] == .6 and r['speed'] == .5)
    row['early_ascent_slip_p95'] *= 2.
    assert low_mu_error(bad, seed) > low_mu_error(improved, seed)
    row['early_ascent_slip_p95'] = None
    assert max(low_mu_errors(bad, seed)) == 4.


def test_actual_bad_64098_cannot_replace_58098(tmp_path):
    from descent_traction_selection import DescentTractionBehaviorSelector
    ref, data = paths()
    current = ROOT/'artifacts/early_traction_58098_20261125_175334_1006334/eval_006000'
    if not current.is_dir():
        pytest.skip('Local measured V1 regression evidence is unavailable')
    checkpoint = ROOT/'logs/rsl_rl/M20-V11.1-EarlyTraction58098/2026-11-25_17-53-45_450637_early_traction_from58098/model_64098.pt'
    if not checkpoint.is_file():
        pytest.skip('Historical model_64098.pt is excluded from this publication')
    teacher = ref/'teacher/model_58098.pt'
    selector = DescentTractionBehaviorSelector(tmp_path, teacher, ref/'native/results.json',
        entry_results=ref/'entry/results.json', flat_reference_results=ref/'flat_start/results.json',
        flat_reference_checkpoint=teacher, forward_reference_results=ref/'production/results.json',
        traction_results=ROOT/'artifacts/route_traction_20261125/reference58098/traction_b1_b3/results.json',
        descent_results=ref/'descent/results.json')
    result = selector.consider(checkpoint, current/'results.json', entry_results=current/'entry/results.json',
        flat_results=current/'flat_start/results.json', forward_results=current/'production/results.json',
        traction_results=current/'traction/results.json', descent_results=current/'quiet_descent/results.json')
    assert not result['promoted'] and not result['descent_retained']
    assert any('low-mu early-tread slip' in r for r in result['regressions'])
    assert Path(result['best_checkpoint']).name == 'model_58098.pt'


def test_material_cohorts_touch_wheels_only_and_are_robot_coherent(monkeypatch):
    from types import ModuleType
    package = ModuleType('_v2_material'); package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    base = ModuleType('isaaclab.envs.mdp.events')
    class MaterialBase:
        def __init__(self, cfg, env):
            self.asset_cfg = cfg.params['asset_cfg']
            self.asset = env.scene['robot']
            self.num_shapes_per_body = [2, 1, 2, 1, 2, 1, 2, 1]
            self.material_buckets = torch.zeros(256, 3)
    base.randomize_rigid_body_material = MaterialBase
    monkeypatch.setitem(sys.modules, base.__name__, base)
    module = load(package.__name__+'.cohorts', SOURCE/'tasks/manager_based/locomotion/velocity/mdp/v111_friction_cohorts.py')
    original = torch.tensor([.9, .9, .1]).expand(20, 12, 3).clone()
    view = NS(get_material_properties=lambda:original.clone())
    result = {}
    view.set_material_properties = lambda data, ids:result.update(data=data, ids=ids)
    class Scene(dict):
        num_envs = 20
    env = NS(scene=Scene(robot=NS(root_physx_view=view)), device='cpu')
    cfg = NS(params={'asset_cfg':NS(body_ids=[1, 3, 5, 7])})
    term = module.WheelFrictionCohorts(cfg, env)
    term(env, None, (.45, .7), (.45, .7), (0., 0.), 256, cfg.params['asset_cfg'], True)
    data = result['data']
    wheel_shapes = [2, 5, 8, 11]
    for body_shape in set(range(12))-set(wheel_shapes):
        assert torch.equal(data[:, body_shape], original[:, body_shape])
    for i in range(20):
        if i % 10 >= 8:
            assert torch.equal(data[i], original[i])
        else:
            values = data[i, wheel_shapes, 0]
            assert (values == values[0]).all()
            assert (.45 <= values[0] <= .7) if i % 10 < 4 else (.85 <= values[0] <= 1.2)
    assert int(env._m20_v2_low_mu.sum()) == 8


def test_descent_route_survives_parent_command_resampling(monkeypatch):
    package = types.ModuleType('_v2_descent_runtime')
    package.__path__ = [str(SOURCE/'tasks/manager_based/locomotion/velocity/mdp')]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    rl = types.ModuleType('rl_training'); rl.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, 'rl_training', rl)
    parent = types.ModuleType(package.__name__+'.v111_route_hold'); parent.reset = lambda *a:None
    flat = types.ModuleType(package.__name__+'.v111_flat_posture'); flat.FlatPostureVelocityCommand = object
    geometry = types.ModuleType(package.__name__+'.v135_terrain_geometry')
    geometry.column_names = lambda cfg:('flat', 'pyramid_stairs', 'pyramid_stairs_inv')
    for m in (parent, flat, geometry):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    module = load(package.__name__+'.v111_descent_traction', Path(package.__path__[0])/'v111_descent_traction.py')
    data = NS(heading_w=torch.zeros(3), root_pos_w=torch.zeros(3, 3),
              root_lin_vel_w=torch.tensor([[.7, 0., 0.]]*3), root_ang_vel_w=torch.zeros(3, 3))
    command = torch.tensor([[.7, 0., 0.]]*3)
    term = NS(is_heading_env=torch.zeros(3, dtype=torch.bool), refine_heading=torch.zeros(3))
    env = NS(scene={'robot':NS(data=data)}, command_manager=NS(get_command=lambda n:command,get_term=lambda n:term),
             cfg=NS(scene=NS(terrain=NS(terrain_generator=None))), device='cpu', num_envs=3,
             episode_length_buf=torch.ones(3, dtype=torch.long), common_step_counter=1)
    class Scene(dict):pass
    env.scene = Scene(env.scene); env.scene.terrain = NS(terrain_types=torch.arange(3))
    module.state(env)
    data.heading_w[1] = -.1; data.root_pos_w[1, 1] = -.15
    env.common_step_counter += 1; env.episode_length_buf += 1
    old = module.state(env)['cost'].clone()
    assert old[1] < 0 and old[0] == old[2] == 0
    term.refine_heading[1] = -.1  # native parent re-sampled at its current heading
    env.common_step_counter += 1; env.episode_length_buf += 1
    assert torch.equal(old, module.state(env)['cost'])
    command[1, 2] = .2; env.common_step_counter += 1
    assert not module.state(env)['active'][1]
