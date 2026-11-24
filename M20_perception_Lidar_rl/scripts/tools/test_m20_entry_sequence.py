"""Counterexamples for the 49998 entry/flat repair and its retention boundary."""
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


math = load('entry_sequence_math_test', SOURCE/'entry_sequence.py')


def test_flat_terms_prefer_progress_with_compact_stance_and_straight_route():
    n = 6
    reach = torch.zeros(n, 4); reach[1] = .30
    yaw = torch.tensor([0., 0., .15, -.15, 0., 0.])
    lateral = torch.tensor([0., 0., .3, -.3, 0., 0.])
    speed = torch.ones(n); speed[4] = 0.
    active = torch.ones(n, dtype=torch.bool); active[5] = False
    r = math.flat_motion_terms(reach, yaw, lateral, speed, torch.ones(n),
        torch.zeros(n), torch.zeros(n), torch.tensor([[.26, -.26, .26, -.26]]).repeat(n, 1), active)
    assert r['quality'][0] > r['quality'][1] > r['quality'][4] == 0.
    assert r['quality'][0] > r['quality'][2] == r['quality'][3]
    assert r['cost'][0] < r['cost'][1] and r['cost'][2] == r['cost'][3]
    assert r['cost'][5] == r['quality'][5] == 0.
    assert (r['cost'] <= .5).all()


def assets():
    reference = ROOT/'artifacts/entry_49998_repair_20261124/reference'
    return dict(teacher=ROOT/'interview_prep/model_49998.pt',
        replay=ROOT/'data/m20_recovery/reference_49998_sequence_20261124.npz',
        native=reference/'native/results.json', entry=reference/'entry/results.json',
        flat=reference/'flat_start/results.json', production=reference/'production/results.json')


def test_physical_replay_has_no_unfinished_b1_and_rejects_tampered_mask(tmp_path):
    from m20_entry_sequence_preflight import validate
    paths = assets()
    if not all(path.is_file() for path in paths.values()): pytest.skip('Collect seed evidence first')
    result = validate(**paths)
    assert result['completed_ascent_reference_cases'] == 8 and result['post_b2_samples'] > 0
    with np.load(paths['replay'], allow_pickle=False) as d:
        data = {k: d[k].copy() for k in d.files}
    index = np.flatnonzero((data['scene'] == 1) & ~data['post_entry'])[0]
    data['post_entry'][index] = True
    changed = tmp_path/'replay.npz'; np.savez_compressed(changed, **data)
    metadata = json.loads(paths['replay'].with_suffix('.json').read_text())
    import hashlib
    metadata['replay_sha256'] = hashlib.sha256(changed.read_bytes()).hexdigest()
    changed.with_suffix('.json').write_text(json.dumps(metadata))
    paths['replay'] = changed
    with pytest.raises(ValueError, match='unfinished B1'):
        validate(**paths)


def test_actual_new_runner_releases_b1_keeps_later_motion_and_recovers_feedback():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    paths = assets()
    if not all(path.is_file() for path in paths.values()): pytest.skip('Collect seed evidence first')
    package = types.ModuleType('_m20_sequence_cpu'); package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    module = load(package.__name__+'.v111_entry_sequence_runner', SOURCE/'v111_entry_sequence_runner.py')
    obs = TensorDict({'policy': torch.zeros(4, 63), 'critic': torch.zeros(4, 568)}, batch_size=[4])
    groups = dict(actor=['policy'], critic=['critic'])
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[512, 256, 128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution',
                          'std_type':'log', 'init_std':1.})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[512, 256, 128])
    alg = module.EntrySequencePPO(actor, critic, None, teacher_checkpoint=str(paths['teacher']),
        replay_path=str(paths['replay']), anchor_weight=2., anchor_front_stair_weight=.04,
        anchor_batch_size=16, behavior_reference_results=str(paths['native']),
        entry_reference_results=str(paths['entry']), flat_reference_results=str(paths['flat']),
        flat_reference_checkpoint=str(paths['teacher']), forward_reference_results=str(paths['production']),
        learning_rate=1.e-5)
    alg.actor.load_state_dict(torch.load(paths['teacher'], map_location='cpu', weights_only=True)['actor_state_dict'])
    before = {k: v.clone() for k, v in alg.actor.state_dict().items()}
    assert all(p.requires_grad for p in alg.actor.parameters())
    assert len(alg.success_members) == 8
    mask = torch.as_tensor(alg._post_b2_mask)
    assert all(mask[group].all() for group in alg.success_members)
    prediction = torch.ones(4, 16, requires_grad=True)
    # All B1 actions, including rear sagittal and wheel drive, are released.
    loss = alg.reference_loss(prediction, torch.zeros_like(prediction), obs['policy'],
        torch.full((4,), 2), torch.ones(4))
    loss.backward()
    assert loss == 0 and not prediction.grad.any()
    # Flat steering and sagittal correction are released while drive remains.
    pred = torch.zeros(4, 16); pred[:, :12] = 2.; pred[:, [12, 14]] = 1.; pred[:, [13, 15]] = -1.
    x = obs['policy'].clone(); x[:, 6] = 1.
    assert alg.reference_loss(pred, torch.zeros_like(pred), x, torch.full((4,), 2), torch.zeros(4)) == 0
    assert alg.support_reference_loss(torch.zeros(4, 16), torch.ones(4, 16)) > 0
    alg._anchor_step(alg.optimizer, (), {})
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.actor.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in alg.actor.parameters())
    assert not alg.optimizer.state
    assert all(torch.equal(v, before[k]) for k, v in alg.actor.state_dict().items())
    for _ in range(30): alg.behavior_feedback(['entry: lost alternating', 'flat: forward_lateral_max_abs'])
    assert alg.learning_rate == 1.e-5 and alg.behavior_anchor_multiplier == 1.
    for _ in range(30): alg.behavior_feedback(['ascent: fell, lost whole_body_top_supported'])
    assert alg.learning_rate == pytest.approx(6.e-6) and alg.behavior_anchor_multiplier == 1.5
    for _ in range(30): alg.behavior_feedback([])
    assert alg.learning_rate == 1.e-5 and alg.behavior_anchor_multiplier == 1.


def test_new_baseline_can_be_archived_and_resumed_with_physical_full_stick_evidence(tmp_path):
    from flat_start_selection import FlatEntryBehaviorSelector
    p = assets()
    if not all(path.is_file() for path in p.values()): pytest.skip('Collect seed evidence first')
    options = dict(flat_reference_results=p['flat'], flat_reference_checkpoint=p['teacher'],
                   forward_reference_results=p['production'])
    selector = FlatEntryBehaviorSelector(tmp_path/'first', p['teacher'], p['native'], p['entry'], **options)
    outcome = selector.consider(p['teacher'], p['native'], p['entry'], p['flat'], p['production'])
    assert not outcome['promoted'] and not outcome['regressions']
    assert not outcome['entry_passed'] and not outcome['production_passed']
    assert (Path(selector.best_checkpoint).parent/'execution_config_origin.json').is_file()
    restored = FlatEntryBehaviorSelector(tmp_path/'resumed', p['teacher'], p['native'], p['entry'],
        previous_best=selector.export_state(), **options)
    assert Path(restored.best_checkpoint).is_file()
    assert restored.forward_rows == selector.forward_rows
    from flat_start_selection import regressions
    rows = json.loads(p['flat'].read_text())
    for key, delta in (('lateral_peak_m', .2), ('yaw_peak_deg', 5.)):
        bad = copy.deepcopy(rows); bad[1][key] += delta
        assert any(key in reason for reason in regressions(bad, rows))
