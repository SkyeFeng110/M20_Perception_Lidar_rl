"""Counterexamples for contact-ordered B1 entry and later gait protection."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/tools'))
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
spec = importlib.util.spec_from_file_location('entry_math', ROOT/'source/rl_training/rl_training/entry_transfer.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from m20_entry_diagnostics import entry_report


def fixture():
    s = m.make_state(1, 'cpu')
    args = dict(x=torch.full((1, 2), -.30), bottom=torch.zeros(1, 2),
        force=torch.full((1, 2), 50.), wheel_vz=torch.zeros(1, 2),
        root_speed=torch.tensor([.5]), front_speed=torch.full((1, 2), .5),
        pitch=torch.zeros(1), pitch_rate=torch.zeros(1), lower=torch.zeros(1),
        rise=torch.tensor([.16]), tread=torch.tensor([.26]), active=torch.tensor([True]),
        reset=torch.tensor([False]), dt=.02,
        rear_x=torch.full((1, 2), -.9), rear_bottom=torch.zeros(1, 2),
        rear_force=torch.full((1, 2), 50.))
    return s, args


def step(s, args, *, side=None, top=False):
    if side is not None:
        args['x'][0, side] = .12 if top else -.05
        args['bottom'][0, side] = .16 if top else .10
        args['force'][0, side] = 50. if top else 0.
    return m.transfer_step(s, **args)


@pytest.mark.parametrize('first', [0, 1])
def test_true_alternation_pays_once_and_has_no_later_cycle_constraint(first):
    s, args = fixture(); step(s, args)
    total = float(step(s, args, side=first)['credit'])
    for _ in range(3): total += float(step(s, args, side=first, top=True)['credit'])
    total += float(step(s, args, side=1-first)['credit'])
    for _ in range(3): total += float(step(s, args, side=1-first, top=True)['credit'])
    assert s['complete'].all() and not s['early'].any() and not s['twin'].any()
    assert 2.2 <= total <= 2.9
    # Later simultaneous legs can move freely; no global alternation clock.
    args['x'][:] = 1.; args['bottom'][:] = .48; args['force'][:] = 0.
    args['rear_x'][:] = .5; args['rear_bottom'][:] = .32
    r = step(s, args)
    assert r['credit'] == 0 and r['cost'] == 0 and not r['near'].any()


def test_second_lift_fails_even_after_long_time_gap_if_first_not_loaded():
    s, args = fixture(); step(s, args); step(s, args, side=0)
    for _ in range(20): step(s, args)
    r = step(s, args, side=1)
    assert r['early_event'].all() and r['twin_event'].all() and r['cost'] > 0


def test_top_wheel_is_not_airborne_and_contact_chatter_cannot_farm_credit():
    s, args = fixture(); step(s, args); step(s, args, side=0)
    result = step(s, args, side=0, top=True)
    assert not result['air'][0, 0]
    # Repeated brief touches never receive stable load credit.
    for _ in range(10):
        step(s, args, side=0, top=False); step(s, args, side=0, top=True)
    assert not s['certified'][0, 0]
    assert float(s['credit_sum']) <= 1.


def test_stopping_cannot_earn_transfer_credit_and_cost_budget_is_finite():
    s, args = fixture()
    args['root_speed'][:] = 0.; args['front_speed'][:] = 0.
    for _ in range(100): assert step(s, args)['credit'] == 0
    args['x'][:] = -.1; args['root_speed'][:] = .5; args['pitch'][:] = .6
    for _ in range(1000): step(s, args)
    assert float(s['cost_used']) <= 4.25001
    for key, cap in (('front_cost_used', 1.5), ('rear_cost_used', 1.),
                     ('posture_cost_used', .75), ('skip_cost_used', 1.)):
        assert float(s[key]) <= cap+1.e-5
    assert float(s['credit_sum']) <= .42501  # .35 approach + .15 * .5 rear hold


def test_rear_lift_is_rejected_far_from_b1_but_allowed_near_b1_and_later():
    s, args = fixture(); step(s, args)
    args['rear_bottom'][0, 0] = .1
    assert step(s, args)['rear_early_event'].all()
    s, args = fixture(); step(s, args)
    args['rear_x'][:] = -.20; args['rear_bottom'][:] = .10
    assert not step(s, args)['rear_early_event'].any()
    args['rear_x'][:] = .5; args['rear_bottom'][:] = .40
    assert not step(s, args)['rear_early_event'].any()


def test_reset_and_starts_on_b1_do_not_invent_an_entry():
    s, args = fixture(); step(s, args); step(s, args, side=0)
    args['reset'][:] = True; args['x'][:] = .1; args['bottom'][:] = .16
    r = step(s, args)
    assert not s['eligible'].any() and r['credit'] == 0 and r['cost'] == 0
    assert s['first'] == -1


def test_early_mate_cannot_get_sequence_bonus_by_delaying_touchdown():
    s, args = fixture(); args['strict'] = True; step(s, args)
    step(s, args, side=0); step(s, args, side=1)
    sequence = 0.
    for _ in range(3): sequence += float(step(s, args, side=0, top=True)['sequence_credit'])
    for _ in range(3): sequence += float(step(s, args, side=1, top=True)['sequence_credit'])
    assert s['complete'].all()  # recovery can continue to later stairs
    assert sequence == 0 and not s['sequence'].any() and not s['clean_complete'].any()
    assert not s['valid_lift'][0, 1]


def test_first_support_must_continue_until_second_is_confirmed():
    s, args = fixture(); step(s, args); step(s, args, side=0)
    for _ in range(3): step(s, args, side=0, top=True)
    step(s, args, side=1)
    r = step(s, args, side=0)
    assert r['front_loss_event'].all() and s['front_loss'].all()
    step(s, args, side=0, top=True)
    for _ in range(3): step(s, args, side=1, top=True)
    assert not s['sequence'].any() and not s['clean_complete'].any()


def test_rear_remains_distance_gated_after_both_front_certificates():
    s, args = fixture(); args['strict'] = True; step(s, args)
    for side in (0, 1):
        step(s, args, side=side)
        for _ in range(3): step(s, args, side=side, top=True)
    assert s['clean_complete'].all() and s['sequence'].all()
    args['rear_bottom'][0, 0] = .1
    r = step(s, args)
    assert r['rear_early_event'].all() and r['cost'] > 0
    args['rear_x'][:] = -.2
    before = s['rear_cost_used'].clone()
    step(s, args)
    assert torch.equal(before, s['rear_cost_used'])


def test_risk_ledgers_do_not_spend_each_others_budget():
    s, args = fixture(); args['strict'] = True; step(s, args)
    step(s, args, side=0); step(s, args, side=1)
    assert s['front_cost_used'] == 1.5
    assert s['rear_cost_used'] == s['posture_cost_used'] == 0.
    args['rear_bottom'][0, 0] = .1
    assert step(s, args)['cost'] > 0 and s['rear_cost_used'] > 0
    args['pitch'][:] = .6
    assert step(s, args)['cost'] > 0 and s['posture_cost_used'] > 0


def test_two_historical_b1_contacts_without_real_lifts_are_not_clean():
    s, args = fixture(); step(s, args)
    for side in (0, 1):
        for _ in range(3): step(s, args, side=side, top=True)
    assert s['complete'].all()
    assert not s['valid_lift'].any() and not s['clean_complete'].any() and not s['sequence'].any()


def test_b1_overshoot_is_charged_once_without_setting_a_speed_limit():
    s, args = fixture(); step(s, args); step(s, args, side=0)
    args['x'][0, 0] = .4; args['bottom'][0, 0] = .32
    r = step(s, args)
    assert r['skip_event'].all() and s['skip_cost_used'] == .5
    before = s['skip_cost_used'].clone()
    for _ in range(20): step(s, args)
    assert torch.equal(before, s['skip_cost_used'])


def arrays():
    n = 35; t = np.arange(n)*.02
    p = np.zeros((n, 4, 3)); p[..., 0] = -.3; p[..., 2] = .09
    labels = np.zeros((n, 4), dtype=int)
    for side, lift, land in ((0, 5, 10), (1, 15, 20)):
        p[lift:, side, 0] = .1
        p[lift:, side, 2] = .25
        labels[lift:land, side] = -1
        labels[land:, side] = 1
        labels[28+side*3:, side] = 2
    root = np.zeros((n, 3)); root[:, 0] = t*.5
    q = np.zeros((n, 4)); q[:, 0] = 1.
    return dict(time=t, wheel_position=p, supported_tread=labels, position=root,
        quaternion=q, angular_velocity=np.zeros((n, 3)))


def test_physics_report_distinguishes_true_alternation_from_twin_jump_and_late_landing():
    d = arrays()
    r = entry_report(d, radius=.09, rise=.16, tread=.26, start_x=0., motion_start=0.)
    assert r['safe_entry'] and r['alternating'] and not r['rear_lift_before_riser']
    # Delaying one touchdown does not make a bilateral jump alternate.
    d['wheel_position'][5:15, 1, 2] = .25; d['supported_tread'][5:15, 1] = -1
    r = entry_report(d, radius=.09, rise=.16, tread=.26, start_x=0., motion_start=0.)
    assert not r['safe_entry'] and not r['alternating'] and r['twin_air_seconds'] > 0


def test_physics_report_rejects_premature_rear_lift_and_missing_entry_scope():
    d = arrays(); d['wheel_position'][6:8, 2, 2] = .25
    r = entry_report(d, radius=.09, rise=.16, tread=.26, start_x=0., motion_start=0.)
    assert r['rear_lift_before_riser'] and not r['safe_entry']
    d = arrays(); d['wheel_position'][:, :2, 0] = .1
    r = entry_report(d, radius=.09, rise=.16, tread=.26, start_x=0., motion_start=0.)
    assert r['scope'] != 'from_lower' and not r['safe_entry']


def test_physics_report_rejects_lead_support_loss_during_handoff():
    d = arrays(); d['supported_tread'][18, 0] = -1
    r = entry_report(d, radius=.09, rise=.16, tread=.26, start_x=0., motion_start=0.)
    assert not r['lead_support_retained'] and not r['alternating'] and not r['safe_entry']
