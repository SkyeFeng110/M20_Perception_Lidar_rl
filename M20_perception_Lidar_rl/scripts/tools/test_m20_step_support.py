"""No-learning regression checks for next-tread reward accounting."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('step_support_math',
    ROOT/'source/rl_training/rl_training/step_support.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    s = dict(target=torch.ones(1, 2).long()*2, peak=torch.zeros(1, 2),
             frames=torch.zeros(1, 2).long())
    def step(*, target=2, x=None, dz=0., force=50., velocity=0., active=True, reset=False, dt=.02):
        x = (target-.5)*.26 if x is None else x
        return m.support_credit_step(s, torch.full((1, 2), target),
            torch.full((1, 2), x), torch.full((1, 2), target*.16+dz),
            torch.full((1, 2), velocity), torch.full((1, 2), force),
            torch.zeros(1), torch.tensor([.16]), torch.tensor([.26]),
            torch.ones(1, 2).bool(), torch.full((1, 2), active), torch.tensor([reset]), dt)
    return s, step


def test_credit_is_distributed_across_true_consecutive_support():
    s, step = fixture()
    values = [step().tolist()[0][0] for _ in range(4)]
    assert values == pytest.approx([.15+.85/3, .85/3, .85/3, 0.])
    assert s['peak'].tolist()[0] == pytest.approx([1., 1.])


def test_airborne_hint_cannot_replace_loaded_support_or_be_farmed():
    _, step = fixture()
    total = 0.
    for _ in range(40):
        total += float(step(force=0., velocity=.5).sum())
        total += float(step(force=0., velocity=-.5).sum())
    assert total == pytest.approx(.30)
    total += sum(float(step().sum()) for _ in range(10))
    assert total == pytest.approx(2.)


@pytest.mark.parametrize('kwargs', [dict(dz=.06), dict(force=0.), dict(force=14.), dict(x=.55)])
def test_wrong_plane_missing_load_and_overshoot_never_earn_dwell(kwargs):
    s, step = fixture()
    for _ in range(10):
        step(**kwargs)
    assert not s['frames'].any()
    assert s['peak'].max() <= .15


def test_interrupted_contact_does_not_count_as_continuous_dwell():
    s, step = fixture()
    step(); step(); step(force=0.); step(); step()
    assert s['frames'].tolist() == [[2, 2]]
    assert s['peak'].max() < 1.
    step()
    assert s['peak'].min() == 1.


def test_same_target_backtracking_inactive_and_holding_never_refill_budget():
    _, step = fixture()
    assert sum(float(step().sum()) for _ in range(3)) == pytest.approx(2.)
    for _ in range(20):
        assert step(x=.1).sum() == 0
        assert step(active=False).sum() == 0
        assert step().sum() == 0


def test_inactive_contact_consumes_credit_without_command_catchup():
    _, step = fixture()
    for _ in range(5):
        assert step(active=False).sum() == 0
    assert step(active=True).sum() == 0


def test_skip_resynchronizes_to_later_target_but_cannot_revisit_old_one():
    s, step = fixture()
    step(force=0., velocity=-.5)
    credit = sum(float(step(target=4).sum()) for _ in range(3))
    assert credit == pytest.approx(2.)
    for _ in range(3):
        assert step(target=2).sum() == 0
    assert s['target'].tolist() == [[4, 4]]


@pytest.mark.parametrize('dt', [.01, .02, .03])
def test_budget_is_policy_rate_independent_and_reset_does_not_pay(dt):
    _, step = fixture()
    assert step(dt=dt, reset=True).sum() == 0
    paid = sum(float(step(dt=dt).sum()) for _ in range(20))
    # Reset on a loaded tread consumes its .15 contact seed.
    assert paid == pytest.approx(1.70)


def test_b1_has_the_same_dense_credit_and_no_lower_surface_fake_support():
    s, step = fixture()
    assert step(target=1, reset=True, dz=-.16).sum() == 0
    assert sum(float(step(target=1).sum()) for _ in range(3)) == pytest.approx(2.)
    assert s['target'].tolist() == [[1, 1]]
