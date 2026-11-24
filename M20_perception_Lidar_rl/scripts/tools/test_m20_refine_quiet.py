"""Behavioral accounting for refinement/noise; no PPO or fitting."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'
MDP = SOURCE/'tasks/manager_based/locomotion/velocity/mdp'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('refine_math', MDP/'v111_refine.py')
reference = load('refine_reference', SOURCE/'recovery_reference.py')
acoustic = load('quiet_mujoco', ROOT.parent/'mujoco_validation/deploy/mujoco_v135/acoustic.py')
guard = load('quiet_guard', ROOT/'scripts/reinforcement_learning/rsl_rl/guarded_recovery.py')
supplement = load('quiet_supplement', ROOT/'scripts/tools/m20_quiet_supplement.py')


def test_landing_credit_cannot_be_farmed_by_holding_revisit_or_closed_command():
    target, peak = torch.zeros(1, 2).long(), torch.zeros(1, 2)
    active = torch.ones(1, 2).bool()
    total = 0
    for expected, score, enabled in [(2, .1, True), (2, .6, True), (2, .2, True),
                                    (2, .6, True), (2, 1., False), (2, 1., True),
                                    (4, .5, True), (4, 1., True)]+[(4, 1., True)]*100:
        target, peak, credit = m.landing_credit(target, peak, torch.full_like(target, expected),
                                               torch.full_like(peak, score), active*enabled)
        total += credit.sum().item()
    assert total == pytest.approx(2.)  # .5 + .5, independently for FL/FR


def test_shape_feedback_survives_bad_yaw_and_requires_real_forward_motion():
    command = torch.tensor([[.7, 0., 0.]]*4+[[.7, 0., .5]])
    velocity = torch.tensor([[.7, .5, 0.], [.7, .5, 0.], [0., 0., 0.], [-.7, 0., 0.], [.7, 0., 0.]])
    value = m.compact_quality(command, velocity, torch.ones(5), torch.tensor([2., 4., 0., 0., 0.]),
                               torch.ones(5), None)
    assert value[0] > value[1] > .1
    assert value[2:].tolist() == [0.]*3


def test_motor_cost_releases_monotonic_lift_and_steady_fast_wheel_drive():
    delta = torch.full((3, 16), .4)
    previous = torch.stack((torch.full((16,), .3), torch.full((16,), -.3), torch.full((16,), -.3)))
    score = m.chatter_score(delta, previous, torch.tensor([False, False, True]))
    assert score[0] == 0
    assert 0 < score[2] < score[1] <= 1
    assert m.chatter_score(torch.zeros_like(delta), previous, torch.zeros(3).bool()).sum() == 0


def test_mean_regularizer_excludes_exploration_and_has_finite_correction_gradient():
    means = torch.stack((torch.zeros(3, 16), torch.full((3, 16), .4), torch.zeros(3, 16))).requires_grad_()
    value = reference.mean_reversal_loss(means, torch.tensor([False, True, False]))
    assert 0 < value <= 1
    value.backward()
    assert torch.isfinite(means.grad).all() and means.grad.abs().sum() > 0
    monotonic = torch.stack((torch.zeros(3, 16), torch.ones(3, 16), torch.full((3, 16), 2.)))
    assert reference.mean_reversal_loss(monotonic, torch.zeros(3).bool()) == 0
    # The API accepts means only; Gaussian std/actions are never an argument.
    assert not any('std' in x for x in reference.mean_reversal_loss.__code__.co_varnames)


def test_mean_regularizer_never_uses_a_triplet_across_reset():
    dones = torch.zeros(6, 2, 1).byte()
    dones[2, 0] = 1
    ids = reference.valid_quiet_triplets(dones).tolist()
    assert [2, 0] in ids and [5, 0] in ids
    assert [3, 0] not in ids and [4, 0] not in ids
    assert all([t, 1] in ids for t in range(2, 6))


def contact_state():
    return {key: torch.zeros(2, 4, dtype=torch.bool if key == 'contact' else torch.float32)
            for key in ('contact', 'air', 'window', 'force', 'velocity')}


def test_contact_hysteresis_no_static_support_tax_and_no_repeat_touchdown():
    s = contact_state()
    reset = torch.ones(2).bool()
    force = torch.full((2, 4), 100.)
    cost, touch = m.contact_step(s, force, torch.zeros_like(force), reset, .02)
    assert cost.sum() == 0 and touch.sum() == 0
    reset[:] = False
    for value in [100.]*20+[10., 6., 10., 100.]:
        cost, touch = m.contact_step(s, force*0+value, force*0, reset, .02)
        assert cost.sum() == 0 and touch.sum() == 0
    for _ in range(4):
        m.contact_step(s, force*0, force*0-.5, reset, .02)
    cost, touch = m.contact_step(s, force, force*0, reset, .02)
    assert (touch == 4).all() and (cost > 0).all()
    for _ in range(10):
        _, touch = m.contact_step(s, force, force*0, reset, .02)
        assert touch.sum() == 0


def test_partial_reset_never_charges_teleport_or_initial_load():
    s = contact_state()
    force = torch.full((2, 4), 100.)
    reset = torch.zeros(2).bool()
    for _ in range(5):
        m.contact_step(s, force*0, force*0-.6, reset, .02)
    cost, touches = m.contact_step(s, force, force*0, torch.tensor([True, False]), .02)
    assert cost[0] == 0 and touches[0] == 0
    assert cost[1] > 0 and touches[1] == 4


def test_noise_cost_budget_and_stage_relative_ramp():
    assert [m.noise_ramp(n*96) for n in (0, 200, 700, 1200, 30000)] == [0., 0., .5, 1., 1.]
    cfg_tree = ast.parse((MDP.parent/'config/wheeled/deeprobotics_m20/v111_refine_env_cfg.py').read_text())
    cfg = next(n for n in cfg_tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20V111RefineCfg')
    values = {n.target.id: ast.literal_eval(n.value) for n in cfg.body if isinstance(n, ast.AnnAssign)}
    assert values['contact_weight'] <= .15
    # Max old flat costs 1.5 + new route .2 + new contact .15 remain smaller
    # than the native velocity gain from rest at all fixed benchmark speeds.
    bound = 1.5+values['route_weight']+values['contact_weight']
    assert (5*(1-torch.exp(-torch.tensor([.5, .7, 1.]).square()/.5)) > bound).all()


def test_roll_contact_velocity_does_not_confuse_hub_travel_with_slip():
    # A forward rolling wheel has nonzero hub speed and zero contact velocity.
    point = acoustic.contact_point_velocity(np.array([1., 0., 0.]),
        np.array([0., 10., 0.]), np.array([0., 0., -.1]))
    np.testing.assert_allclose(point, 0., atol=1e-12)


def test_physical_stair_anchor_allows_idle_correction_but_keeps_common_drive():
    obs, scene = torch.zeros(1, 63), torch.ones(1).long()
    common = torch.zeros(1, 16); common[:, 12:] = 1
    differential = common.clone(); differential[:, 12:] = torch.tensor([1., -1., 1., -1.])
    def loss(x, **kw):
        return reference.reference_anchor_loss(x, torch.zeros_like(x), obs, scene, .04, 2., **kw)
    options = dict(physical_ascent=True, ascent_wheel_differential_weight=.08)
    assert loss(common, **options) == pytest.approx(loss(common).item())
    assert loss(differential, **options) == pytest.approx(.08*loss(common).item())
    legs = torch.zeros(1, 16); legs[:, [1, 2, 4, 5]] = 1
    assert loss(legs, **options) == pytest.approx(.04*loss(legs).item())


def test_descent_failure_is_evidence_but_missing_measurements_are_errors():
    result = dict(status='fallen', goal_reached=False, fell=True,
                  acoustic_proxies=dict(schema_version=1, phases=dict(moving=dict(seconds=.02))))
    supplement.validate_result(result)
    with pytest.raises(RuntimeError):
        supplement.validate_result(dict(result, acoustic_proxies={}))


def test_tensorboard_logs_physical_cases_and_omits_missing_acoustic_evidence():
    row = dict(case='z_c_8/ascent/0.26/0.7', whole_body_top_supported=True, all_treads_supported=False,
               speed=.7, reach_p95=None, acoustic_proxies=dict(phases=dict(moving=dict(
                   touchdown_approach_rms_m_s=None, leg_action_reversal_rms=.2))))
    summary = dict(retention_passed=False, repair_passed=False, cases=[row])
    scalars = guard.evaluation_scalars(summary)
    assert scalars['MuJoCo/ascent_top_cases'] == 1 and scalars['MuJoCo/all_treads_cases'] == 0
    assert not any(k.endswith('reach_p95') or k.endswith('touchdown_approach_rms_m_s') for k in scalars)
    assert scalars['AcousticProxy/z_c_8/ascent/0.26/0.7/moving/leg_action_reversal_rms'] == .2


def test_no_proxy_improvement_claim_by_slower_motion_or_missing_baseline():
    def row(value):
        return dict(acoustic_proxies=dict(phases=dict(moving=dict(touchdown_approach_rms_m_s=value))))
    result = guard.acoustic_comparison(row(.5), row(1.), .7, .7)
    assert result['moving_proxy_ratios']['touchdown_approach_rms_m_s'] == .5
    assert not guard.acoustic_comparison(row(.5), row(1.), .5, .7)['moving_proxy_ratios']
    result = guard.acoustic_comparison(row(.5), {}, .7, .7)
    assert not result['baseline_proxy_available'] and not result['moving_proxy_ratios']


def test_command_cohort_idle_ramp_straight_and_other_commands_preserved():
    # Exercise the actual command overrides, without importing Isaac/Kit.
    tree = ast.parse((MDP/'v111_refine_commands.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and getattr(n, 'name', '') != 'reset_refinement']
    class Base:
        def __init__(self, cfg, env):
            self.num_envs, self.device, self._env = 8, 'cpu', env
            self.robot = NS(data=NS(heading_w=torch.zeros(8), root_pos_w=torch.zeros(8, 3)))
            self.vel_command_b = torch.ones(8, 3)
            self.is_heading_env = torch.ones(8).bool()
            self.is_standing_env = torch.zeros(8).bool()
        def _resample_command(self, ids):
            pass
        def _update_command(self):
            pass
    namespace = dict(torch=torch, UniformThresholdVelocityCommand=Base)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<command_overrides>', 'exec'), namespace)
    term = namespace['RefineVelocityCommand'](None, NS(step_dt=.02))
    term._resample_command(torch.arange(8))
    for _ in range(99):
        term._update_command()
        assert term.vel_command_b[[0, 4]].sum() == 0
    for _ in range(40):
        term._update_command()
    assert (term.vel_command_b[[0, 4], 0] >= .5).all()
    assert term.vel_command_b[[0, 4], 1:].sum() == 0
    assert (term.vel_command_b[[1, 2, 3, 5, 6, 7]] == 1).all()


def test_supplementary_falls_continue_new_profile_without_promoting(tmp_path, monkeypatch):
    fixtures = load('refine_guard_fixture', ROOT/'scripts/tools/test_m20_guarded_session.py')
    # Reuse fake data, never actual PPO. Its helper patches its own guard.
    monkeypatch.setattr(fixtures, 'guard', guard)
    runner, calls, baseline, logs, output = fixtures.fixture(tmp_path, monkeypatch, fail_at=1)
    evaluate = guard.run_evaluation
    def run(command, **kwargs):
        if 'm20_quiet_supplement.py' not in command[1]:
            return evaluate(command, **kwargs)
        path = Path(command[command.index('--output')+1]); path.mkdir()
        (path/'results.json').write_text(json.dumps([dict(case='descent/.26/.7', goal_reached=False,
             fell=True, observed_bypass=False, acoustic_proxies={})]))
        return NS(returncode=0)
    monkeypatch.setattr(guard, 'run_evaluation', run)
    guard.learn_guarded(runner, total_updates=101, segment_updates=100,
                       baseline_path=baseline, output_path=output, log_dir=logs, task=guard.REFINE_TASK)
    assert len(calls) == 2
    status = json.loads((output/'status.json').read_text())
    assert status['status'] == 'completed_with_retention_failures'
    assert status['task'] == guard.REFINE_TASK
