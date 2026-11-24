"""Regression tests for retaining a moving gait while releasing biased steering."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('retention_math_test', SOURCE/'step_retention.py')
selection = load('retention_selection_test', ROOT/'scripts/reinforcement_learning/rsl_rl/behavior_selection.py')
fixtures = load('retention_acceptance_fixture', ROOT/'scripts/tools/test_m20_repair_acceptance.py')


def test_gait_anchor_releases_steering_and_preserves_nonzero_drive():
    teacher = torch.ones(4, 16)*2.
    predicted = teacher.clone()
    predicted[:, [0, 3, 6, 9]] += 3.
    predicted[:, [12, 14]] += 1.
    predicted[:, [13, 15]] -= 1.
    predicted.requires_grad_()
    assert m.successful_support_loss(predicted, teacher) == 0.
    assert m.moving_common_drive_loss(predicted, teacher, torch.ones(4)) == 0.
    # Standing still is not an alternative way to satisfy the moving teacher.
    assert m.successful_support_loss(torch.zeros_like(predicted), teacher) > 0.
    assert m.moving_common_drive_loss(torch.zeros_like(predicted), teacher, torch.ones(4)) > 0.
    bad = predicted.detach().clone(); bad[:, [1, 2, 4, 5]] += .5
    assert m.successful_support_loss(bad, teacher) > 0.


def test_route_reward_prefers_moving_straight_and_keeps_recovery_live():
    speed = torch.tensor([0., -.1, .7, .7, .7, .7])
    yaw = torch.tensor([0., 0., 0., -.25, .25, 0.])
    lateral = torch.tensor([0., 0., 0., -.4, .4, 0.])
    rate = torch.tensor([0., 0., 0., -.1, .1, .15])
    result = m.forward_route_quality(speed, torch.full_like(speed, .7), yaw, lateral, rate, rate)
    assert result[0] == result[1] == 0.
    assert result[2] > result[3] > result[0]
    assert result[3] == result[4]
    assert result[2] > result[5] > 0.
    assert (result <= 1.).all()


def test_local_drift_has_gradient_and_cannot_be_hidden_by_target_variance():
    predicted = torch.ones(4, 16, requires_grad=True)
    target = torch.zeros_like(predicted, requires_grad=True)
    std = torch.ones_like(predicted, requires_grad=True)
    m.local_mean_drift(predicted, target, std).backward()
    assert predicted.grad.abs().sum() > 0
    assert target.grad is None and std.grad is None


@pytest.mark.parametrize('field,value', [('fell', True), ('stall_detected', True),
    ('whole_body_top_supported', False), ('all_treads_supported', False),
    ('forward_yaw_max_change_deg', 20.), ('forward_lateral_max_abs', .6),
    ('later_stairs_route_velocity_x', .05)])
def test_selection_cannot_trade_acquired_behavior_for_reward(field, value):
    rows, _ = fixtures.accepted_rows()
    changed = copy.deepcopy(rows)
    next(r for r in changed if r['terrain'] == 'ascent')[field] = value
    assert selection.regressions(changed, rows)


def evidence(tmp_path, rows, name):
    run = tmp_path/name; run.mkdir()
    ckpt = run/'model_1.pt'; ckpt.write_bytes(name.encode())
    (run/'params').mkdir(); (run/'params/env.yaml').write_text('native: 63\n')
    result = run/'results.json'; result.write_text(json.dumps(rows))
    for p in ('standing', 'keyboard'):
        (run/p).mkdir()
        (run/p/'manifest.json').write_text(json.dumps(dict(physics_enabled=True,
            checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())))
    return ckpt, result


def test_best_is_separate_and_speed_yaw_failure_cannot_replace_it(tmp_path):
    rows, _ = fixtures.accepted_rows()
    seed, results = evidence(tmp_path, rows, 'seed')
    output = tmp_path/'selection'; output.mkdir()
    selector = selection.BehaviorSelector(output, seed, results)
    improved = copy.deepcopy(rows)
    for row in improved:
        row['forward_yaw_max_change_deg'] *= .5
    checkpoint, measured = evidence(tmp_path, improved, 'improved')
    assert selector.consider(checkpoint, measured)['promoted']
    before = (output/'best_candidate.json').read_bytes()
    bad = copy.deepcopy(improved)
    next(r for r in bad if r['terrain'] == 'ascent')['later_stairs_route_velocity_x'] = 0.
    bad_checkpoint, bad_results = evidence(tmp_path, bad, 'failed')
    assert not selector.consider(bad_checkpoint, bad_results)['promoted']
    assert before == (output/'best_candidate.json').read_bytes()
    selected = json.loads(before)
    assert Path(selected['checkpoint']).read_bytes() == checkpoint.read_bytes()
    assert not selected['fully_accepted']


def test_resume_keeps_measured_best_and_case_protection(tmp_path):
    rows, _ = fixtures.accepted_rows()
    seed, results = evidence(tmp_path, rows, 'seed')
    selector = selection.BehaviorSelector(tmp_path/'first', seed, results)
    improved = copy.deepcopy(rows)
    for row in improved:
        row['forward_yaw_max_change_deg'] *= .5
    checkpoint, measured = evidence(tmp_path, improved, 'improved')
    assert selector.consider(checkpoint, measured)['promoted']
    resumed = selection.BehaviorSelector(tmp_path/'continued', seed, results,
                                        previous_best=selector.export_state())
    assert selection.rank(resumed.rows) == selection.rank(improved)
    assert resumed.export_state()['checkpoint'] != str(checkpoint)
    assert Path(resumed.best_checkpoint).read_bytes() == checkpoint.read_bytes()
    bad = copy.deepcopy(rows)
    next(row for row in bad if row['terrain'] == 'ascent')['all_treads_supported'] = False
    bad_checkpoint, bad_results = evidence(tmp_path, bad, 'bad')
    assert not resumed.consider(bad_checkpoint, bad_results)['promoted']


def test_early_physical_failure_with_null_metrics_is_rejected(tmp_path):
    rows, _ = fixtures.accepted_rows()
    seed, results = evidence(tmp_path, rows, 'seed')
    selector = selection.BehaviorSelector(tmp_path/'selection', seed, results)
    failed = copy.deepcopy(rows)
    row = next(r for r in failed if r['terrain'] == 'flat')
    row.update(fell=True, whole_body_top_supported=False, termination='fall',
               progress_windows=0, minimum_route_speed_ratio=None,
               route_mean_velocity_x=None, p95_max_abs_reach=None,
               forward_yaw_max_change_deg=None, forward_lateral_max_abs=None)
    checkpoint, measured = evidence(tmp_path, failed, 'early_failure')
    assert not selector.consider(checkpoint, measured)['promoted']


def test_guard_continues_and_saves_feedback_without_rewriting_evaluated_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace
    guard_tests = load('retention_guard_fixtures', ROOT/'scripts/tools/test_m20_guarded_session.py')
    guard = guard_tests.guard
    runner, calls, baseline, logs, output = guard_tests.fixture(tmp_path, monkeypatch, fail_at=1)
    rows = json.loads(baseline.read_text())
    seed, measured = evidence(tmp_path, rows, 'teacher')
    (logs/'params').mkdir()
    (logs/'params/env.yaml').write_text('native: 63\n')
    alg = SimpleNamespace(behavior_teacher_checkpoint=str(seed), behavior_reference_results=str(measured),
        behavior_selection_state=None, learning_rate=5.e-6, behavior_anchor_multiplier=1.)
    def feedback(regressions):
        if regressions:
            alg.learning_rate *= .8
            alg.behavior_anchor_multiplier *= 1.25
        return dict(learning_rate=alg.learning_rate, success_anchor_multiplier=alg.behavior_anchor_multiplier)
    alg.behavior_feedback = feedback
    runner.alg = alg
    runner.save = lambda path: Path(path).write_text(json.dumps(dict(lr=alg.learning_rate,
        multiplier=alg.behavior_anchor_multiplier, best=alg.behavior_selection_state)))
    original_evaluate = guard.run_evaluation
    def evaluate(command, **kwargs):
        target = Path(command[command.index('--output')+1])
        if 'm20_quiet_supplement.py' in command[1]:
            target.mkdir()
            (target/'results.json').write_text('[]')
            return SimpleNamespace(returncode=0)
        result = original_evaluate(command, **kwargs)
        candidate = Path(command[command.index('--checkpoint')+1])
        for profile in ('standing', 'keyboard'):
            (target/profile).mkdir()
            (target/profile/'manifest.json').write_text(json.dumps(dict(physics_enabled=True,
                checkpoint_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())))
        if len(calls) == 1:
            changed = copy.deepcopy(rows)
            next(r for r in changed if r['terrain'] == 'ascent')['all_treads_supported'] = False
            (target/'results.json').write_text(json.dumps(changed))
        return result
    monkeypatch.setitem(sys.modules, 'behavior_selection', selection)
    monkeypatch.setattr(guard, 'run_evaluation', evaluate)
    monkeypatch.setattr(guard, 'write_evaluation_tensorboard', lambda *args: None)
    guard.learn_guarded(runner, total_updates=200, segment_updates=100, baseline_path=baseline,
                        output_path=output, log_dir=logs, task=guard.STEP_RETENTION_TASK)
    assert len(calls) == 2
    status = json.loads((output/'status.json').read_text())
    assert status['status'] == 'completed_with_retention_failures'
    state = json.loads(Path(status['last_resume_checkpoint']).read_text())
    assert state['lr'] == pytest.approx(4.e-6) and state['multiplier'] == 1.25
    assert Path(state['best']['checkpoint']).read_bytes() == seed.read_bytes()
    assert (logs/'model_35400.pt').read_bytes() == b'fake-test-checkpoint'
    assert (logs/'model_35500.pt').read_bytes() == b'fake-test-checkpoint'


def test_real_rsl_hook_uses_47900_and_has_finite_gradients_without_model_updates(tmp_path, monkeypatch):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    package = types.ModuleType('_m20_retention_cpu')
    package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    runner = load(package.__name__+'.v111_step_retention_runner', SOURCE/'v111_step_retention_runner.py')
    obs = TensorDict({'policy': torch.zeros(4, 63), 'critic': torch.zeros(4, 568)}, batch_size=[4])
    groups = dict(actor=['policy'], critic=['critic'])
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[512, 256, 128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution',
                          'std_type':'log', 'init_std':1.})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[512, 256, 128])
    teacher = ROOT/'logs/rsl_rl/M20-V11.1-StepSupport/2026-11-19_18-47-36_766802_step_support/model_47900.pt'
    replay = ROOT/'data/m20_recovery/reference_47900_20261120.npz'
    if not teacher.exists() or not replay.exists():
        pytest.skip('Local benchmark assets required for the real-checkpoint gradient check')
    alg = runner.StepRetentionPPO(actor, critic, None, teacher_checkpoint=str(teacher), replay_path=str(replay),
        anchor_weight=2., anchor_front_stair_weight=.04, anchor_batch_size=16,
        behavior_reference_results=str(ROOT/'artifacts/step_retention_20261120/reference47900/results.json'))
    alg.actor.load_state_dict(torch.load(teacher, weights_only=True)['actor_state_dict'])
    before = {k:v.clone() for k,v in alg.actor.state_dict().items()}
    assert len(alg.success_members) == 7
    assert all(p.requires_grad for p in alg.actor.parameters())
    assert all(not p.requires_grad for p in alg.teacher.parameters())
    alg._anchor_step(alg.optimizer, (), {})
    assert all(torch.isfinite(p.grad).all() for p in alg.actor.parameters() if p.grad is not None)
    assert all(torch.equal(v,before[k]) for k,v in alg.actor.state_dict().items())
    assert all(p.grad is None for p in alg.teacher.parameters())
    for _ in range(20):
        feedback = alg.behavior_feedback(['lost support'])
    assert feedback['learning_rate'] >= 2.e-6
    assert feedback['success_anchor_multiplier'] == 4.
    assert all(torch.equal(v,before[k]) for k,v in alg.actor.state_dict().items())
    results = ROOT/'artifacts/step_retention_20261120/reference47900/results.json'
    selector = selection.BehaviorSelector(tmp_path/'selected', teacher, results)
    alg.behavior_selection_state = selector.export_state()
    saved = alg.save()
    continuation = tmp_path/'resume_after_eval.pt'
    torch.save(saved, continuation)
    monkeypatch.setitem(sys.modules, 'behavior_selection', selection)
    preflight = load('retention_resume_preflight', ROOT/'scripts/tools/m20_step_retention_preflight.py')
    assert preflight.validate(teacher, replay, results, continuation)['successful_ascent_cases'] == 7
    torch.save({'actor_state_dict': saved['actor_state_dict']}, tmp_path/'missing_state.pt')
    with pytest.raises(ValueError, match='saved StepRetention protection state'):
        preflight.validate(teacher, replay, results, tmp_path/'missing_state.pt')
    alg.behavior_anchor_multiplier = 1.
    alg.behavior_selection_state = None
    alg.load(saved, dict(actor=True, critic=True, optimizer=True, iteration=True), True)
    assert alg.behavior_anchor_multiplier == 4.
    assert alg.behavior_selection_state == selector.export_state()
