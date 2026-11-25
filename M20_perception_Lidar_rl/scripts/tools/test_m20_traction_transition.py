"""Counterexamples for transition sampling and contact-point traction metrics."""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load('m20_traction_math', ROOT/'source/rl_training/rl_training/traction_transition.py')
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
selection = load('m20_traction_selection', ROOT/'scripts/reinforcement_learning/rsl_rl/traction_selection.py')


def test_profiles_cover_partial_steps_and_releases():
    t = torch.tensor([0., 1.999, 2., 4.999, 5., 5.999, 6., 6.6])
    expected = {
        0: [0., 0., .3, .3, .3, .3, .3, .3],
        1: [0., 0., .5, .5, .5, .5, .5, .5],
        2: [0., 0., .7, .7, .7, .7, .7, .7],
        3: [0., 0., 1., 1., 1., 1., .5, .5],
        4: [0., 0., .3, .3, 1., 1., 1., 1.],
        5: [0., 0., 0., 1., 1., 1., 1., 1.],
    }
    for profile, row in expected.items():
        actual = m.transition_speed(t, torch.full_like(t, profile, dtype=torch.long))
        torch.testing.assert_close(actual, torch.tensor(row))


def test_no_slip_rolling_has_zero_residual_and_spin_in_place_does_not():
    linear = torch.zeros(1, 4, 3); linear[..., 0] = 1.
    angular = torch.zeros_like(linear); angular[..., 1] = 10.
    assert torch.allclose(m.rolling_residual(linear, angular, .1), torch.zeros(1, 4, 2))
    linear.zero_()
    torch.testing.assert_close(m.rolling_residual(linear, angular, .1)[..., 0], -torch.ones(1, 4))
    angular.zero_(); linear[..., 0] = .6
    torch.testing.assert_close(m.rolling_residual(linear, angular, .1)[..., 0], torch.full((1, 4), .6))


def test_unloaded_wheels_and_stopping_cannot_improve_traction_reward():
    residual = torch.ones(2, 4, 2)*.5
    supported = torch.tensor([[True, True, True, True], [True, True, False, False]])
    reward, cost, _ = m.traction_terms(residual, supported, torch.full((2,), .6), torch.full((2,), .6))
    assert reward[1] < reward[0] and cost[1] < cost[0]
    assert (.45*reward-.10*cost)[1] < (.45*reward-.10*cost)[0]
    stopped, _, _ = m.traction_terms(residual, supported, torch.zeros(2), torch.full((2,), .6))
    assert torch.equal(stopped, torch.zeros_like(stopped))


def test_early_tread_objective_only_credits_loaded_progress_and_cannot_reward_unloading():
    slip = torch.tensor([[0., .1, .5, .5], [0., .1, .5, .5],
                         [0., .1, .5, .5], [0., .1, .5, .5]])
    supported = torch.tensor([[True, True, True, True], [True, True, False, False],
                              [True, True, True, True], [True, True, True, True]])
    early = torch.tensor([[True, True, True, True], [True, True, True, True],
                          [True, True, False, False], [False, False, False, False]])
    reward, cost, selected, quality = m.early_tread_traction_terms(
        slip, supported, early, torch.tensor([1., 1., 1., 0.]), torch.ones(4))
    assert selected[0].all()
    assert selected[1].tolist() == [True, True, False, False]
    assert selected[2].tolist() == [True, True, False, False]
    combined = .60*reward-.13*cost
    assert combined[0] > combined[1] > 0.
    assert reward[3] == cost[3] == 0.
    assert quality[0, 0] > quality[0, 2]
    # Even nearly unbounded slip leaves a small positive net load credit;
    # unloading cannot beat supported rolling in the actual configured mix.
    severe = torch.full((2, 4), 1000.)
    severe_support = torch.tensor([[True, True, True, True], [False, False, False, False]])
    reward, cost, _, _ = m.early_tread_traction_terms(
        severe, severe_support, torch.ones_like(severe_support), torch.ones(2), torch.ones(2))
    assert (.60*reward-.13*cost)[0] > (.60*reward-.13*cost)[1]


def test_horizontal_patch_rejects_edges_air_impact_slopes_and_invalid_rays():
    hits = torch.zeros(1, 4, 5, 3)
    normals = torch.zeros_like(hits); normals[..., 2] = 1.
    bottom, force, vz = torch.zeros(1, 4), torch.full((1, 4), 30.), torch.zeros(1, 4)
    assert m.horizontal_patch(bottom, hits, normals, force, vz).all()
    edge = hits.clone(); edge[0, 2, 4, 2] = .1
    assert not m.horizontal_patch(bottom, edge, normals, force, vz)[0, 2]
    assert not m.horizontal_patch(bottom, hits, normals, torch.zeros_like(force), vz).any()
    assert not m.horizontal_patch(bottom, hits, normals, force, vz+1).any()
    assert not m.horizontal_patch(bottom, hits, normals*.9, force, vz).any()
    assert not m.horizontal_patch(bottom, hits+torch.inf, normals, force, vz).any()
    assert not m.horizontal_patch(bottom+.2, hits, normals, force, vz).any()


def test_flat_pitch_tolerance_and_symmetry():
    gravity = torch.tensor([[0., 0., -1.], [.5, 0., -.866], [-.5, 0., -.866]])
    costs = m.flat_pitch_cost(gravity, torch.tensor([0., 1., -1.]))
    assert costs[0] == 0. and costs[1] > 0. and costs[1] == costs[2]


def test_drive_regularization_preserves_motion_but_releases_steering_and_wheel_spin():
    teacher = torch.ones(4, 16)
    predicted = teacher.clone()
    predicted[:, [0, 3, 6, 9]] += 5.
    predicted[:, 12:] *= .8
    assert m.drive_band_loss(predicted, teacher, torch.ones(4)).item() == 0.
    stopped = torch.zeros_like(predicted, requires_grad=True)
    loss = m.drive_band_loss(stopped, teacher, torch.ones(4))
    assert loss > 0.
    loss.backward()
    assert stopped.grad[:, 12:].abs().sum() > 0
    leg = predicted.clone(); leg[:, 1] += 1.
    assert m.support_legs_loss(leg, teacher) > m.support_legs_loss(predicted, teacher)


def test_transition_action_cost_targets_impulses_and_keeps_steady_gait_free():
    previous = torch.zeros(3, 16)
    action = previous.clone()
    action[0, 12:] = .1
    action[1, 12:] = 1.0
    action[2, 12:] = 1.0
    cost = m.transition_action_cost(action, previous,
        torch.tensor([1., 1., .5]), torch.tensor([0., 0., .5]),
        straight=torch.ones(3, dtype=torch.bool), entry_near=torch.zeros(3, dtype=torch.bool))
    assert cost[1] > cost[0] > 0. and cost[2] == 0.


def test_entry_order_step_penalizes_same_tick_or_two_tick_front_contact():
    z = torch.zeros(3, 2, dtype=torch.bool)
    support = z.clone(); support[0] = True; support[1, 0] = True; support[2, 1] = True
    result = m.entry_order_step(z, support, z, z, torch.ones(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.long), torch.full((3,), -1), torch.full((3,), -1), torch.zeros(3, dtype=torch.bool))
    assert result['simultaneous'][0] and not result['simultaneous'][1]
    result2 = m.entry_order_step(result['previous_support'],
        torch.tensor([[True, True], [True, True], [False, False]]), result['previous_air'],
        torch.zeros_like(z), torch.ones(3, dtype=torch.bool), result['age'], result['first_side'], result['first_kind'], result['latched'])
    assert result2['simultaneous'][1]


def traction_row():
    return dict(terrain='flat', tread=None, profile='partial_full', speed=1., friction=1.,
        fell=False, bypass=False, termination='duration', rear_slip_p95=.1,
        rear_support_fraction=.8, steady_tracking_ratio=.8, lateral_max_abs_m=.05,
        yaw_max_abs_deg=2., pitch_peak_deg=10., pitch_rate_peak_deg_s=40.)


@pytest.mark.parametrize('changes', [
    {'rear_slip_p95': .4}, {'rear_slip_p95': None}, {'steady_tracking_ratio': .1},
    {'rear_support_fraction': .1}, {'pitch_peak_deg': 18.}, {'pitch_rate_peak_deg_s': 70.},
    {'fell': True}, {'bypass': True}, {'lateral_max_abs_m': .3}, {'yaw_max_abs_deg': 9.},
])
def test_selection_rejects_low_speed_unloading_pitch_and_slip_regressions(changes):
    reference = traction_row(); candidate = dict(reference, **changes)
    assert selection.regressions([candidate], [reference])


def test_selection_accepts_reduced_slip_with_retained_motion():
    reference = traction_row()
    candidate = dict(reference, rear_slip_p95=.05, pitch_peak_deg=8.)
    assert not selection.regressions([candidate], [reference])


def test_selection_rejects_first_three_tread_slip_regression():
    reference = dict(traction_row(), terrain='ascent', tread=.26, profile='hold', speed=1., friction=.6,
        early_ascent_slip_p95=.2, early_ascent_slip_mean=.1)
    candidate = dict(reference, early_ascent_slip_p95=.4)
    assert any('early-tread slip regression' in item
               for item in selection.regressions([candidate], [reference]))


def test_exact_mujoco_contact_metric_distinguishes_rolling_from_spin():
    mujoco = pytest.importorskip('mujoco')
    import numpy as np
    from types import SimpleNamespace
    benchmark = load('traction_benchmark_test', ROOT/'scripts/tools/m20_traction_benchmark.py')
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><option timestep=".001"/><worldbody>
        <geom name="floor" type="plane" size="10 10 1"/>
        <body pos="0 0 .099"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
          <body name="wheel"><joint type="hinge" axis="0 1 0"/>
            <geom type="sphere" size=".1" mass="3" friction="1 .01 .001"/>
          </body>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    robot = SimpleNamespace(model=model, data=data, terrain_geoms={0},
                            wheels=np.array([model.body('wheel').id, -1, -2, -3]))
    data.qvel[0], data.qvel[6] = 1., 10.
    mujoco.mj_forward(model, data)
    rolling, supported, early_rolling, early_supported = benchmark.loaded_contact_slip(
        robot, (-.01, .01), return_early=True)
    assert supported[0] and rolling[0] < .02
    assert early_supported[0] and early_rolling[0] < .02
    data.qvel[0] = 0.
    mujoco.mj_forward(model, data)
    spin, supported = benchmark.loaded_contact_slip(robot)
    assert supported[0] and spin[0] > .9


def test_real_traction_runner_gradients_preserve_loaded_weights():
    import types
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    source = ROOT/'source/rl_training/rl_training'
    package = types.ModuleType('_m20_traction_cpu')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    runner = load(package.__name__+'.v111_traction_transition_runner', source/'v111_traction_transition_runner.py')
    obs = TensorDict({'policy': torch.zeros(4, 63), 'critic': torch.zeros(4, 568)}, batch_size=[4])
    groups = dict(actor=['policy'], critic=['critic'])
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[512, 256, 128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution',
                          'std_type':'log', 'init_std':1.})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[512, 256, 128])
    teacher = ROOT/'logs/rsl_rl/M20-V11.1-StepSupport/2026-11-19_18-47-36_766802_step_support/model_47900.pt'
    replay = ROOT/'data/m20_recovery/reference_47900_20261120.npz'
    if not teacher.exists() or not replay.exists():
        pytest.skip('Local benchmark assets required')
    alg = runner.TractionTransitionPPO(actor, critic, None, teacher_checkpoint=str(teacher), replay_path=str(replay),
        anchor_weight=2., anchor_front_stair_weight=.04, anchor_batch_size=16,
        behavior_reference_results=str(ROOT/'artifacts/step_retention_20261120/reference47900/results.json'),
        traction_reference_results=str(ROOT/'artifacts/traction_transition_20261120/reference47900/results.json'))
    alg.actor.load_state_dict(torch.load(teacher, weights_only=True)['actor_state_dict'])
    before = {k:v.clone() for k,v in alg.actor.state_dict().items()}
    alg._anchor_step(alg.optimizer, (), {})
    assert all(torch.isfinite(p.grad).all() for p in alg.actor.parameters() if p.grad is not None)
    assert any(p.grad.abs().sum() > 0 for p in alg.actor.parameters() if p.grad is not None)
    assert all(torch.equal(v, before[k]) for k,v in alg.actor.state_dict().items())
    assert all(p.requires_grad for p in alg.actor.parameters())
    assert all(not p.requires_grad for p in alg.teacher.parameters())
    assert not alg.optimizer.state


def test_guard_runs_both_gates_and_keeps_traction_failure_out_of_repaired_status(tmp_path, monkeypatch):
    import json
    import hashlib
    from types import SimpleNamespace
    helpers = load('traction_native_test_helpers', ROOT/'scripts/tools/test_m20_step_retention.py')
    guards = load('traction_guard_test_helpers', ROOT/'scripts/tools/test_m20_guarded_session.py')
    guard = guards.guard
    runner, calls, baseline, logs, output = guards.fixture(tmp_path, monkeypatch, fail_at=1)
    rows = json.loads(baseline.read_text())
    seed, native = helpers.evidence(tmp_path, rows, 'teacher')
    (logs/'params').mkdir(); (logs/'params/env.yaml').write_text('native: 63\n')
    reference = ROOT/'artifacts/traction_transition_20261120/reference47900'
    if not (reference/'results.json').exists():
        pytest.skip('Local physical traction suite required')
    traction_rows = json.loads((reference/'results.json').read_text())
    manifest = json.loads((reference/'manifest.json').read_text())
    for row in traction_rows:
        if row['terrain'] == 'ascent':
            row['early_ascent_slip_p95'] = row['rear_slip_p95']
            row['early_ascent_slip_mean'] = row['rear_slip_mean']
            row['early_loaded_contact_samples'] = row['rear_loaded_samples']
    root = tmp_path/'traction_seed'; root.mkdir()
    manifest['checkpoint_sha256'] = hashlib.sha256(seed.read_bytes()).hexdigest()
    manifest.setdefault('early_tread_window', 'actual contact x in B1-B3')
    manifest.setdefault('early_slip_method', 'test fixture')
    (root/'manifest.json').write_text(json.dumps(manifest))
    (root/'results.json').write_text(json.dumps(traction_rows))
    alg = SimpleNamespace(behavior_teacher_checkpoint=str(seed), behavior_reference_results=str(native),
        behavior_selection_state=None, traction_reference_results=str(root/'results.json'))
    alg.behavior_feedback = lambda failures: {'regression_count': len(failures)}
    runner.alg = alg
    runner.save = lambda path: Path(path).write_bytes(b'continuation')
    original = guard.run_evaluation
    count = 0
    def evaluate(command, **kwargs):
        nonlocal count
        target = Path(command[command.index('--output')+1])
        if 'm20_quiet_supplement.py' in command[1]:
            target.mkdir(); (target/'results.json').write_text('[]')
            return SimpleNamespace(returncode=0)
        if 'm20_traction_benchmark.py' in command[1]:
            target.mkdir(); changed = copy.deepcopy(traction_rows)
            if count == 0:
                changed[0]['rear_slip_p95'] += 1.
            count += 1
            candidate = Path(command[command.index('--checkpoint')+1])
            measured = dict(manifest, checkpoint_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
            (target/'manifest.json').write_text(json.dumps(measured))
            (target/'results.json').write_text(json.dumps(changed))
            return SimpleNamespace(returncode=0)
        result = original(command, **kwargs)
        candidate = Path(command[command.index('--checkpoint')+1])
        for profile in ('standing', 'keyboard'):
            (target/profile).mkdir()
            (target/profile/'manifest.json').write_text(json.dumps(dict(physics_enabled=True,
                checkpoint_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())))
        return result
    monkeypatch.setattr(guard, 'run_evaluation', evaluate)
    monkeypatch.setattr(guard, 'write_evaluation_tensorboard', lambda *args: None)
    monkeypatch.setitem(sys.modules, 'traction_selection', selection)
    guard.learn_guarded(runner, total_updates=200, segment_updates=100,
        baseline_path=baseline, output_path=output, log_dir=logs, task=guard.TRACTION_TRANSITION_TASK)
    status = json.loads((output/'status.json').read_text())
    first = json.loads((output/'eval_000100/behavior_selection.json').read_text())
    assert first['selection']['traction_passed'] is False
    assert count == 2 and len(calls) == 2
    assert status['traction_passed'] is True and status['traction_repair_passed'] is False
    assert status['repair_passed'] is False
    assert 'last_repair_passed_checkpoint' not in status
    assert Path(status['best_candidate_checkpoint']).read_bytes() == seed.read_bytes()
