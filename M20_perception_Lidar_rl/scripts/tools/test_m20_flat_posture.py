"""Counterexamples for full-stick coverage, stance incentives and gait retention."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'
sys.path.insert(0, str(ROOT/'scripts/tools'))
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


math = load('flat_posture_test_math', SOURCE/'flat_posture.py')


def paths():
    r = ROOT/'artifacts/flat_posture_repair_20261124/reference50498'
    d = ROOT/'data/m20_recovery/flat_posture_20261124'
    teacher = r/'teacher/model_50498.pt'
    if not teacher.is_file():
        teacher = ROOT/'logs/rsl_rl/M20-V11.1-EntrySequenceRepair/2026-11-24_13-27-41_476901_sequence_flat_from49998/model_50498.pt'
    return dict(checkpoint=teacher, teacher=teacher,
        replay=d/'current_gait.npz', compact=d/'compact_flat.npz', native=r/'native/results.json',
        entry=r/'entry/results.json', flat=r/'flat_start/results.json', production=r/'production/results.json')


def test_nominal_and_randomized_resets_both_see_direct_and_partial_full_commands():
    ids = torch.arange(64)
    focus, nominal, initial = math.profiles(ids, torch.zeros(64))
    _, _, speed = math.profiles(ids, torch.full((64,), 1.5))
    _, _, later = math.profiles(ids, torch.full((64,), 3.))
    assert not initial.any()
    for cohort in (nominal, focus & ~nominal):
        assert (cohort & (speed == 1.)).any()
        assert (cohort & (speed == .3) & (later == 1.)).any()
        assert (cohort & (speed == .3) & (later == .3)).any()


def test_flat_quality_requires_progress_and_penalizes_one_or_both_extended_rears():
    reach = torch.zeros(5, 4); reach[1, 2] = -.24; reach[2, 2:] = -.24
    active = torch.ones(5, dtype=torch.bool); active[4] = False
    speed = torch.ones(5); speed[3] = 0
    r = math.posture_terms(reach, torch.zeros(5), torch.zeros(5), speed, torch.ones(5),
        torch.zeros(5), torch.zeros(5), torch.tensor([[.26, -.26, .26, -.26]]).repeat(5, 1), active)
    assert r['quality'][0] > r['quality'][1] > r['quality'][2] > r['quality'][3] == 0
    assert r['cost'][0] < r['cost'][1] < r['cost'][2]
    assert r['cost'][4] == r['quality'][4] == 0
    assert (r['cost'] <= 2.).all()


def test_onset_cost_has_independent_finite_budgets_and_no_stair_charge():
    n = 2; window = torch.zeros(n); peaks = torch.zeros(n, 4)
    action = torch.full((n, 16), 8.); previous = torch.zeros_like(action)
    wheel = torch.tensor([[.10, -.10, .25, -.25]]).repeat(n, 1)
    active = torch.tensor([True, False]); reset = torch.zeros(n, dtype=torch.bool)
    window, peaks, cost = math.onset_step(window, peaks, torch.zeros(n), torch.ones(n),
        action, previous, torch.full((n,), 2.), wheel, active, reset, .02)
    assert 0 < cost[0]*.02 <= 2. and cost[1] == 0
    spent = cost*.02
    for _ in range(40):
        window, peaks, cost = math.onset_step(window, peaks, torch.ones(n), torch.ones(n),
            action, previous, torch.full((n,), 4.), wheel, active, reset, .02)
        spent += cost*.02
    assert spent[0] <= 2.000001 and spent[1] == 0
    # A separate command event reopens a budget, not an unlimited per-step tax.
    _, _, cost = math.onset_step(window, peaks, torch.full((n,), .3), torch.ones(n),
        action, previous, torch.full((n,), 2.), wheel, active, reset, .02)
    assert cost[0] > 0 and cost[1] == 0


def test_onset_does_not_charge_small_useful_actions_or_a_reset():
    zeros = torch.zeros(2); action = torch.full((2, 16), .1)
    _, _, cost = math.onset_step(zeros, torch.zeros(2, 4), zeros, torch.ones(2), action,
        torch.zeros_like(action), zeros, torch.tensor([[.26, -.26, .26, -.26]]).repeat(2, 1),
        torch.ones(2, dtype=torch.bool), torch.tensor([False, True]), .02)
    assert not cost.any()


def test_measured_compact_flat_and_full_entry_gait_assets_are_verified():
    from m20_flat_posture_preflight import validate
    p = paths()
    if not all(v.is_file() for v in p.values()): pytest.skip('Collect physical references')
    r = validate(**p)
    assert r['gait_samples'] > 3000 and r['compact_samples'] > 400


def test_entry_b1_and_lead_order_cannot_be_traded_for_better_flat_score():
    from flat_posture_selection import entry_retention
    rows = json.loads(paths()['native'].read_text())
    broken = copy.deepcopy(rows)
    row = next(r for r in broken if r['terrain'] == 'ascent')
    row['first_supported_tread'][0] = 2
    assert any('lost first B1' in f for f in entry_retention(broken, rows))
    broken = copy.deepcopy(rows); row = next(r for r in broken if r['terrain'] == 'ascent')
    row['entry_transfer']['lead_side'] = 1
    assert any('changed lead side' in f for f in entry_retention(broken, rows))


def test_flat_first_archive_and_resume_do_not_require_new_stair_sequence():
    from flat_posture_selection import FlatPostureBehaviorSelector
    import tempfile
    p = paths()
    options = dict(entry_results=p['entry'], flat_reference_results=p['flat'],
        flat_reference_checkpoint=p['teacher'], forward_reference_results=p['production'])
    with tempfile.TemporaryDirectory() as d:
        s = FlatPostureBehaviorSelector(Path(d)/'first', p['teacher'], p['native'], **options)
        result = s.consider(p['teacher'], p['native'], p['entry'], p['flat'], p['production'])
        assert not result['promoted'] and not result['regressions']
        restored = FlatPostureBehaviorSelector(Path(d)/'next', p['teacher'], p['native'],
            previous_best=s.export_state(), **options)
        assert restored.forward_rows == s.forward_rows
        native = json.loads(p['native'].read_text())
        flat = json.loads(p['flat'].read_text()); forward = json.loads(p['production'].read_text())
        entry = json.loads(p['entry'].read_text())
        slow = copy.deepcopy(flat)
        next(r for r in slow if r['case']['name'] == 'raw_full')['final_speed_m_s'] *= .9
        assert any('speed traded for posture' in f for f in s._regressions(native, entry, slow, forward))


def test_valid_early_fall_with_no_measurement_is_rejected_without_crashing_score():
    from flat_posture_selection import posture_score, production_passed
    p = paths(); native = json.loads(p['native'].read_text())
    flat = json.loads(p['flat'].read_text()); forward = json.loads(p['production'].read_text())
    row = next(r for r in forward if r['terrain'] == 'flat')
    row.update(fell=True, p95_max_abs_reach=None, route_mean_velocity_x=None)
    assert np.isfinite(posture_score(native, flat, forward))
    assert not production_passed(forward)


def test_cpu_real_ppo_wiring_all_layers_and_both_replay_gradients(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    p = paths()
    package = types.ModuleType('_m20_posture_cpu'); package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    module = load(package.__name__+'.v111_flat_posture_runner', SOURCE/'v111_flat_posture_runner.py')
    contract = load('flat_posture_joint_limit_contract_test', SOURCE/'joint_target_limits.py').joint_target_contract()
    obs = TensorDict({'policy': torch.zeros(4, 63), 'critic': torch.zeros(4, 568)}, batch_size=[4])
    groups = dict(actor=['policy'], critic=['critic'])
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[512, 256, 128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution', 'std_type':'log', 'init_std':1.})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[512, 256, 128])
    alg = module.FlatPosturePPO(actor, critic, None, teacher_checkpoint=str(p['teacher']),
        replay_path=str(p['replay']), anchor_weight=2., anchor_front_stair_weight=.04,
        anchor_batch_size=32, behavior_reference_results=str(p['native']),
        entry_reference_results=str(p['entry']), flat_reference_results=str(p['flat']),
        flat_reference_checkpoint=str(p['teacher']), forward_reference_results=str(p['production']),
        compact_replay=str(p['compact']), learning_rate=1.e-5, joint_target_contract=contract)
    saved = torch.load(p['teacher'], map_location='cpu', weights_only=True)
    alg.actor.load_state_dict(saved['actor_state_dict'])
    # Real API and gradient wiring only. No optimizer step or physical claim.
    before = {k: v.clone() for k, v in alg.actor.state_dict().items()}
    assert all(p.requires_grad for p in alg.actor.parameters())
    assert len(alg.success_members) == 8
    alg._anchor_step(alg.optimizer, (), {})
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.actor.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in alg.actor.parameters())
    assert alg.compact_losses[-1] > 0
    assert not alg.optimizer.state
    assert all(torch.equal(v, before[k]) for k, v in alg.actor.state_dict().items())
    assert alg.save()['m20_joint_target_limits'] == contract
    with pytest.raises(ValueError, match='initial-checkpoint'):
        alg.load(saved)
    for _ in range(30): alg.behavior_feedback(['flat workspace', 'entry: changed lead side'])
    assert alg.learning_rate == 1.e-5


def test_reward_runtime_covers_unseen_far_approach_but_releases_entry_and_stairs(monkeypatch):
    from types import SimpleNamespace as NS
    mdp = SOURCE/'tasks/manager_based/locomotion/velocity/mdp'
    package = types.ModuleType('_m20_posture_mdp'); package.__path__ = [str(mdp)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    entry = types.ModuleType(package.__name__+'.v111_entry_transfer')
    entry.EntrySequenceVelocityCommand = object; entry.state = lambda env: None
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    geometry = types.ModuleType(package.__name__+'.v135_terrain_geometry')
    geometry.column_names = lambda cfg: ('flat', 'ascent')
    geometry.v135_terrain_geometry = lambda env: {'ascent': env.scene.terrain.terrain_types == 1}
    monkeypatch.setitem(sys.modules, geometry.__name__, geometry)
    monkeypatch.setitem(sys.modules, 'rl_training', types.ModuleType('rl_training'))
    monkeypatch.setitem(sys.modules, 'rl_training.flat_posture', math)
    fake_math = types.ModuleType('isaaclab.utils.math')
    fake_math.quat_apply_inverse = lambda q, v: v  # fixture quaternions are identity
    monkeypatch.setitem(sys.modules, 'isaaclab', types.ModuleType('isaaclab'))
    monkeypatch.setitem(sys.modules, 'isaaclab.utils', types.ModuleType('isaaclab.utils'))
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.math', fake_math)
    module = load(package.__name__+'.v111_flat_posture', mdp/'v111_flat_posture.py')
    n = 5
    root = torch.tensor([[-1., 0., .52], [-1., 0., .52], [-.7, 0., .52], [0., 0., .52], [-1., 0., .52]])
    relative = torch.tensor([[.3, .26, .09], [.3, -.26, .09], [-.3, .26, .09], [-.3, -.26, .09]])
    wheel = relative[None].repeat(n, 1, 1); wheel[..., :2] += root[:, None, :2]
    wheel[4, :, 2] = .30
    hips = wheel.clone(); hips[:, :, 2] = .5; hips[:, 2:, 0] += .24
    positions = torch.cat((wheel, hips), 1)
    names = [leg+'_hipy' for leg in ('fl', 'fr', 'hl', 'hr')]
    data = NS(body_link_pos_w=positions, root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_pos_w=root, heading_w=torch.zeros(n), root_lin_vel_w=torch.tensor([[1., 0., 0.]]).repeat(n, 1),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        root_ang_vel_b=torch.tensor([[0., 2., 0.]]).repeat(n, 1))
    robot = NS(data=data, find_bodies=lambda name: ([4+names.index(name)], [name]))
    class Scene(dict): pass
    scene = Scene(robot=robot); scene.env_origins = torch.zeros(n, 3)
    scene.terrain = NS(terrain_types=torch.tensor([0, 1, 1, 1, 1]))
    command = torch.zeros(n, 3)
    term = NS(vel_command_b=command, is_heading_env=torch.zeros(n, dtype=torch.bool),
        refine_heading=torch.zeros(n), refine_origin=root[:, :2].clone())
    env = NS(num_envs=n, device='cpu', common_step_counter=1, episode_length_buf=torch.ones(n, dtype=torch.long),
        scene=scene, step_dt=.02, cfg=NS(scene=NS(terrain=NS(terrain_generator=None)),
        v11_perception=NS(wheel_radius=.09)), _v111_recovery_wheels=[0, 1, 2, 3],
        _v111_first_tread={'anchor':torch.zeros(n, 2), 'forward':torch.tensor([[1., 0.]]).repeat(n, 1),
            'height_valid':torch.zeros(n, dtype=torch.bool), 'lower':torch.zeros(n)},
        action_manager=NS(action=torch.zeros(n, 16), prev_action=torch.zeros(n, 16)),
        command_manager=NS(get_term=lambda name:term, get_command=lambda name:command))
    module.state(env)
    env.common_step_counter += 1; env.episode_length_buf += 1
    command[:, 0] = 1.; env.action_manager.action[:] = 8.
    s = module.state(env)
    assert s['active'].tolist() == [True, True, False, False, False]
    assert (s['onset_cost'][:2] > 0).all() and not s['onset_cost'][2:].any()
    assert (module.cost(env)[:2] < 0).all() and not module.cost(env)[2:].any()
    # Logging keeps the trajectory's peak after the final frame relaxes.
    old_peak = module.metric(env, torch.arange(n), 'pitch_rate_peak')
    env.common_step_counter += 1; env.episode_length_buf += 1
    data.root_ang_vel_b[:] = 0
    module.state(env)
    assert module.metric(env, torch.arange(n), 'pitch_rate_peak') == old_peak


def test_new_guarded_task_continues_after_valid_behavior_rejection(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import shutil
    guard = load('posture_guard_test', ROOT/'scripts/reinforcement_learning/rsl_rl/guarded_recovery.py')
    p = paths(); monkeypatch.setenv('M20_RECOVERY_TEACHER', str(p['teacher']))
    logs = tmp_path/'run'; logs.mkdir(); shutil.copytree(p['teacher'].parent/'params', logs/'params')
    runner = NS(current_learning_iteration=50498, cfg={}, logger=NS(writer=None))
    runner.alg = NS(entry_reference_results=str(p['entry']), flat_reference_results=str(p['flat']),
        flat_reference_checkpoint=str(p['teacher']), forward_reference_results=str(p['production']),
        behavior_teacher_checkpoint=str(p['teacher']), behavior_reference_results=str(p['native']),
        behavior_selection_state=None, behavior_feedback=lambda failures:dict(learning_rate=1.e-5, success_anchor_multiplier=1.))
    calls = []
    def learn(num_learning_iterations, init_at_random_ep_len):
        calls.append(num_learning_iterations)
        runner.current_learning_iteration += num_learning_iterations-1
        shutil.copy2(p['teacher'], logs/f'model_{runner.current_learning_iteration}.pt')
    runner.learn = learn; runner.save = lambda path:Path(path).write_text('test continuation')
    monkeypatch.setattr(guard, 'resource_check', lambda *args:{})
    monkeypatch.setattr(guard, 'write_evaluation_tensorboard', lambda *args:None)
    def evaluate(command, **kwargs):
        target = Path(command[command.index('--output')+1])
        source = (p['entry'].parent if 'm20_entry_flat_benchmark.py' in command[1] else
            p['flat'].parent if 'm20_flat_command_probe.py' in command[1] else
            p['production'].parent if 'm20_production_transition_benchmark.py' in command[1] else p['native'].parent)
        shutil.copytree(source, target)
        if source == p['native'].parent:
            (target/'acceptance.json').write_text(json.dumps(dict(retention_passed=False, repair_passed=False,
                valid_baseline=True, validation_failures=[], retention_failures=['test retained but strict posture rejected'],
                candidates=[dict(results=str(target/'results.json'))])))
            return NS(returncode=2)
        return NS(returncode=0)
    monkeypatch.setattr(guard, 'run_evaluation', evaluate)
    guard.learn_guarded(runner, total_updates=200, segment_updates=100,
        baseline_path=p['native'], output_path=tmp_path/'session', log_dir=logs,
        task=guard.FLAT_POSTURE_TASK, evaluation_every_updates=100, evaluation_profile='entry')
    status = json.loads((tmp_path/'session/status.json').read_text())
    assert calls == [100, 100] and status['completed_updates'] == 200
    assert status['status'] == 'completed_with_retention_failures'
