"""Behavioral counterexamples for flat reward and stair preservation."""
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


math = load('flat_stability_test_math', SOURCE/'flat_stability.py')


def evaluate(reach, speed=None, flat=None, pitch=None, rate=None, straight=None):
    n = len(reach)
    one, zero = torch.ones(n), torch.zeros(n)
    yes = torch.ones(n, dtype=torch.bool)
    return math.stability_terms(reach, zero, zero, one if speed is None else speed,
        one, zero if pitch is None else pitch, zero if rate is None else rate,
        torch.tensor([[.26, -.26, .26, -.26]]).repeat(n, 1),
        yes if flat is None else flat, yes if straight is None else straight)


def test_opposite_extensions_do_not_cancel_and_extreme_leg_is_not_diluted():
    reach = torch.tensor([[0., 0., 0., 0.], [.15, .15, -.25, -.25], [0., 0., -.25, 0.]])
    r = evaluate(reach)
    assert r['cost'][1] > r['cost'][2] > .3 > r['cost'][0]
    assert r['quality'][0] > r['quality'][2] > r['quality'][1]


def test_no_flat_shaping_or_gradient_on_stairs_even_with_zero_perception():
    reach = torch.full((4, 4), .30, requires_grad=True)
    r = evaluate(reach, flat=torch.tensor([True, False, False, True]))
    (r['cost']-.75*r['quality']).sum().backward()
    assert not r['cost'][1:3].any() and not r['quality'][1:3].any()
    assert not reach.grad[1:3].any() and reach.grad[0].abs().sum() > 0


def test_cost_stays_corrective_at_measured_reach_and_bounded_at_extremes():
    reach = torch.tensor([[.20]*4, [.25]*4, [.30]*4, [10.]*4], requires_grad=True)
    r = evaluate(reach, pitch=torch.tensor([0., 0., 0., 10.]), rate=torch.tensor([0., 0., 0., 100.]))
    r['cost'].sum().backward()
    assert (reach.grad[:3] > 0).all()
    assert (torch.diff(r['cost']) > 0).all() and (r['cost'] <= math.COST_CAP).all()


def test_progress_remains_profitable_and_standing_earns_no_quality():
    speed = torch.tensor([0., .3, .7, 1.])
    r = evaluate(torch.full((4, 4), .25), speed=speed)
    score = .75*r['quality']-r['cost']
    assert r['quality'][0] == 0 and (torch.diff(score) > 0).all()


def test_turning_does_not_disable_stance_or_reward_old_route():
    r = evaluate(torch.full((2, 4), .25), straight=torch.tensor([True, False]))
    assert r['cost'][1] > 0 and r['quality'][1] == 0


def test_stair_target_retains_all_actions_without_freezing_any_layer():
    pred = torch.ones(3, 16, requires_grad=True)
    target = torch.zeros_like(pred, requires_grad=True)
    math.stair_action_loss(pred, target).backward()
    assert (pred.grad > 0).all() and target.grad is None
    assert math.stair_action_loss(torch.full((1,16), .02), torch.zeros(1,16)) == 0


def test_measured_baseline_passes_and_new_descent_failure_is_rejected():
    from m20_flat_stability_acceptance import stair_regressions
    base = ROOT/'artifacts/flat_50498_20261124'
    up = json.loads((base/'ascent/results.json').read_text())
    down = json.loads((base/'descent/results.json').read_text())
    assert not stair_regressions(up, down, up, down)
    bad = copy.deepcopy(down); bad[0]['goal_reached'] = False
    assert stair_regressions(up, bad, up, down)
    bad_up = copy.deepcopy(up)
    next(r for r in bad_up if r['terrain'] == 'ascent')['skipped_treads'][0].append(99)
    assert any('more skipped' in r for r in stair_regressions(bad_up, down, up, down))


def test_explicit_assets_work_without_historical_default(monkeypatch, tmp_path):
    assets = load('recovery_assets_stability_test', SOURCE/'recovery_assets.py')
    monkeypatch.setattr(assets, 'DATA', tmp_path)
    monkeypatch.setenv('M20_RECOVERY_TEACHER', str(tmp_path/'teacher.pt'))
    monkeypatch.setenv('M20_RECOVERY_REPLAY', str(tmp_path/'replay.npz'))
    assert assets.resolve_assets() == (tmp_path/'teacher.pt', tmp_path/'replay.npz')


def test_actual_runner_gradient_and_resume_identity():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from test_m20_entry_sequence import assets
    package = types.ModuleType('_m20_stability_cpu'); package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    module = load(package.__name__+'.v111_flat_stability_runner', SOURCE/'v111_flat_stability_runner.py')
    p = assets()
    obs = TensorDict({'policy': torch.zeros(4,63), 'critic': torch.zeros(4,568)}, batch_size=[4])
    groups = dict(actor=['policy'], critic=['critic'])
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[512,256,128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution', 'std_type':'log','init_std':1.})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[512,256,128])
    alg = module.FlatStabilityPPO(actor, critic, None, teacher_checkpoint=str(p['teacher']),
        replay_path=str(p['replay']), anchor_weight=2., anchor_front_stair_weight=.04, anchor_batch_size=32,
        behavior_reference_results=str(p['native']), entry_reference_results=str(p['entry']),
        flat_reference_results=str(p['flat']), flat_reference_checkpoint=str(p['teacher']),
        forward_reference_results=str(p['production']),
        stair_replay_path=str(ROOT/'data/m20_recovery/stairs_50498_stability_20261124.npz'), learning_rate=1.e-5)
    assert len(alg.stair_members) == 10 and all(p.requires_grad for p in actor.parameters())
    before = {k:v.clone() for k,v in actor.state_dict().items()}
    alg._anchor_step(alg.optimizer, (), {})
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    assert not alg.optimizer.state and all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
    prediction = torch.ones(4,16,requires_grad=True)
    assert alg.reference_loss(prediction, prediction*0, obs['policy'], torch.ones(4), torch.zeros(4)) == 0
    assert alg.support_reference_loss(prediction, prediction*0) == 0
    with pytest.raises(ValueError, match='Resume requires'):
        alg.load({'m20_flat_stability': {}}, {'optimizer': True})
