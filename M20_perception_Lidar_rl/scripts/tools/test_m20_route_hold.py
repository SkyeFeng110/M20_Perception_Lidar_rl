"""Counterexamples for persistent route learning and acquired ability retention."""
import copy
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
sys.path.insert(0, str(ROOT/'scripts/tools'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


math = load('_route_hold_test_math', SOURCE/'route_hold.py')


def paths():
    r = ROOT/'artifacts/route_hold_20261125/reference56898'
    d = ROOT/'data/m20_recovery/route_hold_20261125'
    return dict(checkpoint=r/'teacher/model_56898.pt', teacher=r/'teacher/model_56898.pt',
        replay=d/'current_gait.npz', compact=d/'compact_flat.npz', route_flat=d/'route_flat.npz',
        native=r/'native/results.json', entry=r/'entry/results.json', flat=r/'flat_start/results.json',
        production=r/'production/results.json',
        traction_results=ROOT/'artifacts/route_traction_20261125/reference58098/traction_b1_b3/results.json',
        traction_teacher=ROOT/'logs/rsl_rl/M20-V11.1-RouteHold/2026-11-25_11-30-18_303338_route_hold_latest/model_58098.pt')


def test_constant_wrong_heading_and_world_side_speed_are_not_stable_straight_motion():
    # Zero body vy/yaw rate can coexist with 5deg heading, 40cm line drift
    # and world-route side speed. Stopping must not remove the route cost.
    r = math.route_terms(torch.tensor([0., .0873, .0873, -.0873, .0873]),
        torch.tensor([0., .4, .4, -.4, .4]), torch.tensor([0., .075, 0., -.075, .075]),
        torch.zeros(5), torch.tensor([1., 1., 0., 1., 1.]), torch.ones(5),
        torch.tensor([True, True, True, True, False]), torch.ones(5, dtype=torch.bool))
    assert r['cost'][1] < r['cost'][0] == 0
    assert r['quality'][0] > r['quality'][1] > r['quality'][2] == 0
    assert r['cost'][2] < r['cost'][1]
    assert r['cost'][1] == r['cost'][3] and r['quality'][1] == r['quality'][3]
    assert r['cost'][4] == r['quality'][4] == 0


def test_correction_toward_original_heading_and_line_has_room_to_act():
    r = math.route_terms(torch.full((2,), .1), torch.full((2,), .1),
        torch.tensor([.1, -.1]), torch.tensor([.2, -.2]), torch.ones(2), torch.ones(2),
        torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    assert r['cost'][1] > r['cost'][0]


def test_route_reference_survives_command_resampling_and_idle_but_resets_after_turn():
    heading, origin, valid = torch.zeros(1), torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool)
    def sample(yaw, x, straight=True, reset=False):
        nonlocal heading, origin, valid
        heading, origin, valid = math.latch_reference(heading, origin, valid,
            torch.tensor([yaw]), torch.tensor([[x, .1*x]]),
            torch.tensor([straight]), torch.tensor([reset]))
    sample(0., 0.)
    sample(.1, 1.); sample(.2, 2.)  # resample at displaced poses, including idle
    assert heading.item() == origin.abs().sum().item() == 0
    sample(.3, 3., straight=False); assert not valid.item()
    sample(.4, 4.); assert heading.item() == pytest.approx(.4)
    sample(0., 0., reset=True); assert heading.item() == origin.abs().sum().item() == 0


def test_equivariance_releases_asymmetric_gait_and_common_forward_drive(monkeypatch):
    package = types.ModuleType('_m20_route_equivariance'); package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    m = load(package.__name__+'.route_hold', SOURCE/'route_hold.py')
    ref = load(package.__name__+'.targeted_reference', SOURCE/'targeted_reference.py')
    # Different sagittal leg actions are allowed; reflected input should
    # produce the reflected gait, not equalize both legs in one state.
    action = torch.randn(8, 16)
    mirrored = ref.mirror_action(action)
    assert m.steering_equivariance(action, mirrored).item() == 0
    mirrored[:, 12:] += 3.
    assert m.steering_equivariance(action, mirrored).item() < 1.e-12
    mirrored[:, 0] += .5
    assert m.steering_equivariance(action, mirrored).item() > 0
    masked = ref.masked_equivariance_loss(action, mirrored, torch.tensor([True, False]*4))
    selected = torch.tensor([True, False]*4)
    expected = ref.equivariance_loss(action[selected], mirrored[selected])
    assert torch.allclose(masked, expected)
    assert ref.masked_equivariance_loss(action, mirrored, torch.zeros(8, dtype=torch.bool)).item() == 0
    masked_route = m.masked_steering_equivariance(action, mirrored, selected)
    expected_route = m.steering_equivariance(action[selected], mirrored[selected])
    assert torch.allclose(masked_route, expected_route)
    assert m.masked_steering_equivariance(action, mirrored, torch.zeros(8, dtype=torch.bool)).item() == 0
    support = load('_route_hold_support_math', SOURCE/'step_retention.py')
    changed = action.clone(); changed[:, [0, 3, 6, 9]] += 1.
    changed[:, 12:] += torch.tensor([1., -1., 1., -1.])
    assert support.successful_support_loss(changed, action).item() == 0


def test_runtime_route_projection_reference_and_whole_episode_metrics(monkeypatch):
    mdp = SOURCE/'tasks/manager_based/locomotion/velocity/mdp'
    package = types.ModuleType('_m20_route_mdp'); package.__path__ = [str(mdp)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, 'rl_training', types.ModuleType('rl_training'))
    monkeypatch.setitem(sys.modules, 'rl_training.route_hold', math)
    geo = types.ModuleType(package.__name__+'.v135_terrain_geometry')
    geo.column_names = lambda cfg: ('flat', 'ascent', 'descent')
    geo.ascent_name = lambda name: name == 'ascent'
    monkeypatch.setitem(sys.modules, geo.__name__, geo)
    parent = types.ModuleType(package.__name__+'.v111_flat_posture')
    parent.reset = lambda env, ids: None
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    m = load(package.__name__+'.v111_route_hold', mdp/'v111_route_hold.py')
    n = 4
    data = NS(heading_w=torch.zeros(n), root_pos_w=torch.zeros(n, 3),
        root_lin_vel_w=torch.tensor([[1., 0., 0.]]).repeat(n, 1), root_ang_vel_w=torch.zeros(n, 3))
    class Scene(dict): pass
    scene = Scene(robot=NS(data=data)); scene.terrain = NS(terrain_types=torch.tensor([0, 1, 1, 2]))
    command = torch.tensor([[1., 0., 0.]]).repeat(n, 1); command[2, 2] = .4
    term = NS(is_heading_env=torch.zeros(n, dtype=torch.bool), refine_heading=torch.zeros(n), refine_origin=torch.zeros(n, 2))
    env = NS(num_envs=n, device='cpu', common_step_counter=1, episode_length_buf=torch.ones(n, dtype=torch.long),
        cfg=NS(scene=NS(terrain=NS(terrain_generator=None))), scene=scene,
        command_manager=NS(get_command=lambda name:command, get_term=lambda name:term))
    m.state(env)
    data.heading_w[:] = .1; data.root_pos_w[:, 1] = .3; data.root_lin_vel_w[:, 1] = .1
    # This is precisely the old reference reset that used to erase the error.
    term.refine_heading[:] = .1; term.refine_origin[:, 1] = .3
    env.common_step_counter += 1; env.episode_length_buf += 1
    s = m.state(env)
    assert s['active'].tolist() == [True, True, False, False]
    assert m.metric(env, [1], 'ascent_yaw') == pytest.approx(.1)
    assert m.metric(env, [1], 'ascent_lateral_speed') == pytest.approx(.1)
    assert m.metric(env, [1], 'ascent_samples') == 1
    m.state(env); assert m.metric(env, [1], 'ascent_samples') == 1
    data.root_pos_w[:, 1] = 0
    env.common_step_counter += 1; env.episode_length_buf += 1
    assert m.metric(env, [1], 'ascent_lateral_peak') == pytest.approx(.3)
    assert m.metric(env, [1], 'ascent_lateral') == pytest.approx(.15)
    original = lambda env: torch.ones(n)
    assert m.outside_focus_reward(env, original, {}).tolist() == [0., 0., 1., 1.]
    parent.state = lambda env:dict(shape=torch.full((n,), .2), attitude=torch.full((n,), .3),
        active=torch.ones(n, dtype=torch.bool), onset_cost=torch.full((n,), .5),
        cost=torch.full((n,), 100.), quality=torch.zeros(n))
    # Parent resampled-route penalty cannot leak back into posture reward.
    assert torch.allclose(m.flat_posture_cost(env), torch.full((n,), -.92))
    assert (m.flat_posture_quality(env) > 0).all()
    m.reset(env, [1]); env.common_step_counter += 1; env.episode_length_buf[1] = 0
    assert m.metric(env, [1], 'ascent_samples') == 0


def test_route_ppo_factory_binds_the_unwrapped_reward_environment(monkeypatch):
    package = types.ModuleType('_m20_route_factory'); package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load(package.__name__+'.v111_route_hold_runner', SOURCE/'v111_route_hold_runner.py')
    reward_env, algorithm = NS(), NS(early_traction_multiplier=1.15)
    algorithm._sync_traction_feedback = lambda: setattr(
        reward_env, '_m20_route_early_traction_multiplier', algorithm.early_traction_multiplier)
    with monkeypatch.context() as patch:
        patch.setattr(module.FlatPosturePPO, 'construct_algorithm', staticmethod(lambda *args: algorithm))
        result = module.RouteHoldPPO.construct_algorithm(None, NS(unwrapped=reward_env), {}, 'cpu')
    assert result is algorithm
    assert reward_env._m20_route_early_traction_multiplier == 1.15


def test_measured_56898_assets_and_executed_flat_actions_are_verified():
    from m20_route_hold_preflight import validate
    result = validate(**paths(), joint_limits=True)
    assert result['completed_ascent_cases'] == 8
    assert result['route_flat_samples'] == 750 and result['gait_samples'] > 2500


def test_environment_replaces_old_route_terms_and_keeps_speed_posture_and_limits():
    # Execute the new config's real post_init on the saved parent reward
    # graph, without importing the simulator or constructing an Isaac app.
    import yaml
    p = paths()
    saved = yaml.load((p['teacher'].parent/'params/env.yaml').read_text(), Loader=yaml.UnsafeLoader)
    path = SOURCE/'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v111_route_hold_env_cfg.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        and n.name == 'DeeproboticsM20V111RouteHoldEnvCfg')
    route = NS(**{name: object() for name in ('cost', 'quality', 'flat_posture_cost', 'flat_posture_quality',
                                            'reset', 'metric', 'outside_focus_reward', 'early_traction_quality',
                                            'early_traction_cost', 'early_traction_metric')})
    class Parent:
        def __post_init__(self): pass
    scope = dict(DeeproboticsM20V111FlatPostureEnvCfg=Parent, configclass=lambda c:c,
                 RewTerm=NS, CurrTerm=NS, route=route)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), scope)
    cfg = scope[cls.name]()
    cfg.rewards = NS(**{name:(NS(**value) if value else None) for name,value in saved['rewards'].items()})
    cfg.events = NS(v111_refine_reset=NS(func=None)); cfg.curriculum = NS()
    before = copy.deepcopy(vars(cfg.rewards))
    cfg.__post_init__()
    for name in ('path_lateral_deviation', 'path_heading_error', 'stair_straightness',
                 'v111_refine_route', 'v111_retention_forward_route', 'v111_straight_stability'):
        term = getattr(cfg.rewards, name)
        assert term.func is route.outside_focus_reward and term.weight == before[name].weight
        assert term.params == dict(reward_func=before[name].func, reward_params=before[name].params)
    assert cfg.rewards.v111_flat_posture_cost.func is route.flat_posture_cost
    for name in ('world_forward_progress', 'world_forward_deficit', 'track_lin_vel_xy_exp',
                 'v111_joint_target_limit', 'v111_recovery_progress'):
        assert vars(getattr(cfg.rewards, name)) == vars(before[name])
    assert cfg.rewards.v111_entry_overshoot_speed is None
    assert cfg.rewards.v111_entry_sequence is None
    assert cfg.rewards.v111_route_early_traction_quality.weight == .60
    assert cfg.rewards.v111_route_early_traction_cost.weight == .13
    assert cfg.curriculum.v111_route_early_traction_slip_mean.params['key'] == 'slip_mean'
    assert cfg.curriculum.v111_route_hold_ascent_lateral_sum.params['key'] == 'ascent_lateral_sum'


def selector_options(p):
    return dict(entry_results=p['entry'], flat_reference_results=p['flat'],
        flat_reference_checkpoint=p['teacher'], forward_reference_results=p['production'],
        traction_results=p['traction_results'])


def traction_candidate(tmp_path, source, checkpoint, *, improved=False):
    target = tmp_path/('traction_improved' if improved else 'traction_candidate')
    target.mkdir()
    import hashlib
    manifest = json.loads((Path(source).parent/'manifest.json').read_text())
    manifest['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    rows = json.loads(Path(source).read_text())
    if improved:
        for row in rows:
            if row['terrain'] == 'ascent':
                row['early_ascent_slip_p95'] *= .5
                row['early_ascent_slip_mean'] *= .5
    (target/'manifest.json').write_text(json.dumps(manifest))
    (target/'results.json').write_text(json.dumps(rows))
    return target/'results.json'


def test_biased_seed_not_certified_and_route_improvement_is_selected_with_retention(tmp_path):
    from route_hold_selection import RouteHoldBehaviorSelector
    p = paths()
    s = RouteHoldBehaviorSelector(tmp_path/'first', p['teacher'], p['native'], **selector_options(p))
    traction = traction_candidate(tmp_path, p['traction_results'], p['teacher'])
    initial = s.consider(p['teacher'], p['native'], p['entry'], p['flat'], p['production'], traction)
    assert not initial['promoted'] and not initial['regressions']
    assert initial['route_unresolved'] and not initial['route_hold_passed']
    assert initial['feedback_failures']  # unresolved defects also drive feedback
    changed = {}
    for name in ('native', 'entry', 'flat', 'production'):
        target = tmp_path/name
        shutil.copytree(p[name].parent, target)
        rows = json.loads(p[name].read_text())
        for row in rows:
            fields = ('yaw_peak_deg', 'lateral_peak_m') if name == 'flat' else ('forward_yaw_max_change_deg', 'forward_lateral_max_abs')
            for field in fields: row[field] *= .5
        (target/'results.json').write_text(json.dumps(rows))
        changed[name] = target/'results.json'
    changed_traction = traction_candidate(tmp_path, p['traction_results'], p['teacher'], improved=True)
    improved = s.consider(p['teacher'], changed['native'], changed['entry'], changed['flat'], changed['production'], changed_traction)
    assert improved['promoted'] and not improved['regressions']
    assert improved['best_rank'] > initial['best_rank']
    restored = RouteHoldBehaviorSelector(tmp_path/'next', p['teacher'], p['native'],
        previous_best=s.export_state(), **selector_options(p))
    assert restored.forward_rows == s.forward_rows
    # A better route score cannot buy extended flat legs or loss of ascent.
    native = json.loads(changed['native'].read_text()); entry = json.loads(changed['entry'].read_text())
    flat = json.loads(changed['flat'].read_text()); forward = json.loads(changed['production'].read_text())
    next(r for r in forward if r['terrain'] == 'flat')['p95_max_abs_reach'] += .05
    assert any('workspace' in f for f in s._regressions(native, entry, flat, forward))
    next(r for r in native if r['terrain'] == 'ascent')['whole_body_top_supported'] = False
    assert any('lost whole_body_top_supported' in f for f in s._regressions(native, entry, flat, forward))


def test_real_cpu_ppo_gradients_full_actor_resume_and_feedback_bounds(tmp_path, monkeypatch):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    package = types.ModuleType('_m20_route_ppo'); package.__path__ = [str(SOURCE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load(package.__name__+'.v111_route_hold_runner', SOURCE/'v111_route_hold_runner.py')
    contract = load('_route_target_contract', SOURCE/'joint_target_limits.py').joint_target_contract()
    p = paths()
    obs = TensorDict({'policy': torch.zeros(4, 63), 'critic': torch.zeros(4, 568)}, batch_size=[4])
    groups = dict(actor=['policy'], critic=['critic'])
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[512, 256, 128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution', 'std_type':'log', 'init_std':1.})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[512, 256, 128])
    alg = module.RouteHoldPPO(actor, critic, None, teacher_checkpoint=str(p['teacher']),
        replay_path=str(p['replay']), anchor_weight=2., anchor_front_stair_weight=.04,
        anchor_batch_size=32, behavior_reference_results=str(p['native']),
        entry_reference_results=str(p['entry']), flat_reference_results=str(p['flat']),
        flat_reference_checkpoint=str(p['teacher']), forward_reference_results=str(p['production']),
        compact_replay=str(p['compact']), route_flat_replay=str(p['route_flat']),
        traction_reference_results=str(p['traction_results']),
        learning_rate=8.e-6, joint_target_contract=contract)
    reward_env = NS()
    alg._reward_env = reward_env
    alg._sync_traction_feedback()
    assert reward_env._m20_route_early_traction_multiplier == 1.
    saved = torch.load(p['teacher'], map_location='cpu', weights_only=True)
    alg.load(saved, dict(actor=True, critic=True, optimizer=False))
    assert all(parameter.requires_grad for parameter in alg.actor.parameters())
    before = {k:v.clone() for k,v in alg.actor.state_dict().items()}
    assert len(alg.success_members) == 8
    alg._anchor_step(alg.optimizer, (), {})
    assert alg.route_steering_losses[-1] > 0 and np.isfinite(alg.route_flat_losses[-1])
    assert alg.success_member_table.shape[0] == len(alg.success_member_sizes) == len(alg.success_members)
    choices = torch.randint(len(alg.success_members), (64,))
    offsets = (torch.rand(64)*alg.success_member_sizes[choices]).long()
    samples = alg.success_member_table[choices, offsets]
    assert all(samples[i] in alg.success_members[int(choices[i])] for i in range(len(samples)))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.actor.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in alg.actor.parameters())
    assert not alg.optimizer.state
    assert all(torch.equal(v, before[k]) for k,v in alg.actor.state_dict().items())
    for _ in range(30): feedback = alg.behavior_feedback(['ascent: route target forward_lateral_max_abs'])
    assert alg.learning_rate == 8.e-6 and alg.behavior_anchor_multiplier == 1.
    assert alg.route_steering_weight == .5 and feedback['action'] == 'record_and_continue'
    for _ in range(4): feedback = alg.behavior_feedback(['traction/ascent: early-tread slip regression'])
    assert 1.5 < alg.early_traction_multiplier <= 1.6
    assert reward_env._m20_route_early_traction_multiplier == alg.early_traction_multiplier
    for _ in range(30): alg.behavior_feedback([])
    assert alg.early_traction_multiplier == 1.
    alg.behavior_feedback(['current_z_c_8/flat/None/1.0: flat workspace'])
    assert alg.route_flat_multiplier > 1.
    for _ in range(30): alg.behavior_feedback(['flat/raw_full: pitch_peak_deg', 'ascent: fell'])
    assert alg.learning_rate == pytest.approx(.6*8.e-6)
    assert alg.route_flat_multiplier == 1.5 and alg.behavior_anchor_multiplier == 1.5
    state = alg.save(); alg.load(state)
    assert reward_env._m20_route_early_traction_multiplier == alg.early_traction_multiplier
    for _ in range(30): alg.behavior_feedback([])
    assert alg.learning_rate == 8.e-6 and alg.route_steering_weight == .15 and alg.route_flat_multiplier == 1.
    with pytest.raises(ValueError, match='fresh optimizer'): alg.load(saved)
    state['m20_route_hold']['flat_replay_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='fresh optimizer'): alg.load(state)


def test_new_guard_does_not_stop_for_unresolved_route_or_valid_behavior_rejection(tmp_path, monkeypatch):
    guard = load('_route_guard_test', ROOT/'scripts/reinforcement_learning/rsl_rl/guarded_recovery.py')
    p = paths(); monkeypatch.setenv('M20_RECOVERY_TEACHER', str(p['teacher']))
    logs = tmp_path/'run'; logs.mkdir(); shutil.copytree(p['teacher'].parent/'params', logs/'params')
    runner = NS(current_learning_iteration=56898, cfg={}, logger=NS(writer=None))
    feedbacks = []
    def feedback(failures):
        feedbacks.append(failures)
        return dict(learning_rate=8.e-6, success_anchor_multiplier=1., route_steering_weight=.15,
                    route_flat_multiplier=1., early_traction_multiplier=1., action='record_and_continue')
    runner.alg = NS(entry_reference_results=str(p['entry']), flat_reference_results=str(p['flat']),
        flat_reference_checkpoint=str(p['teacher']), forward_reference_results=str(p['production']),
        behavior_teacher_checkpoint=str(p['teacher']), behavior_reference_results=str(p['native']),
        behavior_selection_state=None, behavior_feedback=feedback,
        traction_reference_results=str(p['traction_results']))
    calls = []
    def learn(num_learning_iterations, init_at_random_ep_len):
        calls.append(num_learning_iterations)
        runner.current_learning_iteration += num_learning_iterations-1
        shutil.copy2(p['teacher'], logs/f'model_{runner.current_learning_iteration}.pt')
    runner.learn = learn; runner.save = lambda path:Path(path).write_text('fake continuation')
    monkeypatch.setattr(guard, 'resource_check', lambda *args:{})
    summaries = []
    monkeypatch.setattr(guard, 'write_evaluation_tensorboard', lambda s,*args:summaries.append(s))
    def evaluate(command, **kwargs):
        target = Path(command[command.index('--output')+1])
        source = (p['traction_results'].parent if 'm20_traction_benchmark.py' in command[1] else
            p['entry'].parent if 'm20_entry_flat_benchmark.py' in command[1] else
            p['flat'].parent if 'm20_flat_command_probe.py' in command[1] else
            p['production'].parent if 'm20_production_transition_benchmark.py' in command[1] else p['native'].parent)
        shutil.copytree(source, target)
        if source == p['traction_results'].parent:
            import hashlib
            manifest = json.loads((target/'manifest.json').read_text())
            manifest['checkpoint_sha256'] = hashlib.sha256(p['teacher'].read_bytes()).hexdigest()
            (target/'manifest.json').write_text(json.dumps(manifest))
        if source == p['native'].parent:
            (target/'acceptance.json').write_text(json.dumps(dict(retention_passed=False, repair_passed=True,
                valid_baseline=True, validation_failures=[], retention_failures=['test strict rejection'],
                candidates=[dict(results=str(target/'results.json'))])))
            return NS(returncode=2)
        return NS(returncode=0)
    monkeypatch.setattr(guard, 'run_evaluation', evaluate)
    guard.learn_guarded(runner, total_updates=200, segment_updates=100,
        baseline_path=p['native'], output_path=tmp_path/'session', log_dir=logs,
        task=guard.ROUTE_HOLD_TASK, evaluation_every_updates=100, evaluation_profile='entry')
    status = json.loads((tmp_path/'session/status.json').read_text())
    assert calls == [100, 100] and status['completed_updates'] == 200
    assert status['status'] == 'completed_with_retention_failures'
    assert not status['route_hold_passed'] and not status['repair_passed']
    assert len(feedbacks) == 2 and all(any('route target' in f for f in fs) for fs in feedbacks)
    tags = guard.evaluation_scalars(summaries[-1])
    assert tags['RouteHold/passed'] == 0 and 'MuJoCo/best_candidate_all_treads_cases' not in tags
