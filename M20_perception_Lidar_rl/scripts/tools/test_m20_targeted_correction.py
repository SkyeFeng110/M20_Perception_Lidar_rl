"""Targeted correction gradients, phase budgets, kinematic reflection and guards."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'source/rl_training/rl_training'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('targeted_math', SOURCE/'targeted_reference.py')
reference = load('targeted_anchor', SOURCE/'recovery_reference.py')


def test_mirror_native_abi_involution_and_forward_edge_order():
    x = torch.randn(11, 63)
    torch.testing.assert_close(m.mirror_observation(m.mirror_observation(x)), x)
    torch.testing.assert_close(m.mirror_observation(x)[:, 57:], x[:, 57:])
    a = torch.randn(11, 16)
    torch.testing.assert_close(m.mirror_action(m.mirror_action(a)), a)
    assert m.mirror_observation(x).shape == (11, 63)


def test_targeted_contract_rejects_changed_joint_or_observation_scales():
    nominal = [0.,-.3,.6]*2+[0.,.3,-.6]*2+[0.]*4
    scale = [.125,.25,.25]*4+[5.]*4
    observations = dict(base_ang_vel=.25, projected_gravity=1., velocity_commands=1.,
        joint_pos=1., joint_vel=.05, actions=1., height_scan=1.)
    m.validate_execution_contract(nominal, scale, observations)
    with pytest.raises(ValueError, match='verified native'):
        m.validate_execution_contract(nominal, [1.]*16, observations)
    with pytest.raises(ValueError, match='verified native'):
        m.validate_execution_contract(nominal, scale, dict(observations, joint_vel=1.))


def test_reflection_matches_real_mujoco_leg_kinematics():
    import mujoco
    path = ROOT.parent/'mujoco_validation/deploy/mujoco_v135/assets/m20/mjcf/M20.xml'
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    legs = ('fl', 'fr', 'hl', 'hr')
    names = [f'{leg}_{joint}_joint' for leg in legs for joint in ('hipx', 'hipy', 'knee')]
    names += [f'{leg}_wheel_joint' for leg in legs]
    qids = [model.jnt_qposadr[model.joint(n).id] for n in names]
    bodies = [model.body(leg+'_wheel').id for leg in legs]
    nominal = np.array([0, -.3, .6]*2+[0, .3, -.6]*2+[0]*4)
    for seed in range(4):
        offset = np.random.default_rng(seed).uniform(-.15, .15, 16)
        data.qpos[qids] = nominal+offset
        mujoco.mj_forward(model, data)
        original = data.xpos[bodies].copy()
        data.qpos[qids] = nominal+m.mirror_action(torch.tensor(offset)).numpy()
        mujoco.mj_forward(model, data)
        expected = original[[1, 0, 3, 2]]*np.array([1, -1, 1])
        np.testing.assert_allclose(data.xpos[bodies], expected, atol=2.e-6)
        hip_bodies = [model.body(leg+'_hipy').id for leg in legs]
        action = np.zeros(16)
        action[:12] = offset[:12]/np.array([.125,.25,.25]*4)
        data.qpos[qids] = nominal+offset
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(m.commanded_reach(torch.tensor(action[None])).numpy()[0],
            (data.xpos[bodies]-data.xpos[hip_bodies])[:, 0], atol=2.e-6)


def test_symmetry_preserves_a_valid_asymmetric_gait_and_has_bias_gradient():
    action = torch.randn(5, 16, requires_grad=True)
    assert m.equivariance_loss(action, m.mirror_action(action)) == 0
    biased = torch.zeros(5, 16, requires_grad=True)
    with torch.no_grad():
        biased[:, 12:] = torch.tensor([1., -1., 1., -1.])
    loss = m.equivariance_loss(biased, biased)
    loss.backward()
    assert torch.isfinite(biased.grad).all() and biased.grad[:, 12:].abs().sum() > 0


def test_idle_scope_does_not_suppress_moving_or_perturbed_stair_actions():
    x = torch.zeros(7, 63); x[:, 5] = -1
    phase = torch.ones(7).long()
    x[1, 6] = .7; x[2, 8] = .2; x[3, 5] = -.9
    x[4, 57] = 1; x[5, 9] = .4; phase[6] = 2
    mask = m.settled_idle_mask(x, phase)
    assert mask.tolist() == [True]+[False]*6
    action = torch.ones(7, 16, requires_grad=True)
    m.idle_wheel_loss(action, mask).backward()
    assert action.grad[:, :12].count_nonzero() == 0
    assert action.grad[1:].count_nonzero() == 0
    assert action.grad[0, 12:].abs().sum() > 0


def test_retention_release_keeps_moving_common_drive_and_stair_swing_options():
    x = torch.zeros(2, 63); x[:, 5] = -1; x[1, 6] = .7
    legs, wheels = m.correction_scales(x, torch.ones(2).long())
    predicted = torch.ones(2, 16, requires_grad=True)
    kwargs = dict(physical_ascent=True, ascent_wheel_differential_weight=.08)
    def loss(**more):
        return reference.reference_anchor_loss(predicted, torch.zeros_like(predicted), x,
            torch.ones(2).long(), .04, 2., **kwargs, **more)
    original = torch.autograd.grad(loss(), predicted, retain_graph=True)[0]
    modified = torch.autograd.grad(loss(leg_scale=legs, wheel_scale=wheels), predicted)[0]
    torch.testing.assert_close(modified[1, 12:], original[1, 12:])
    torch.testing.assert_close(modified[0, 12:], .08*original[0, 12:])
    torch.testing.assert_close(modified[:, list(m.HIP)], .10*original[:, list(m.HIP)])
    torch.testing.assert_close(modified[:, [1,2,4,5,7,8,10,11]], original[:, [1,2,4,5,7,8,10,11]])


def test_flat_command_workspace_gradient_has_no_stair_or_wheel_amplitude_cost():
    x = torch.zeros(3, 63); x[:,6] = .7; x[2,6] = 0
    actions = torch.zeros(3,16, requires_grad=True)
    with torch.no_grad():
        actions[:,[1,4,7,10]] = 3.
    value = m.flat_workspace_loss(actions, x, torch.tensor([0,1,0]))
    assert 0 < value <= 1
    value.backward()
    assert actions.grad[0,:12].abs().sum() > 0
    assert actions.grad[:,12:].count_nonzero() == 0
    assert actions.grad[1:].count_nonzero() == 0
    assert m.flat_workspace_loss(torch.zeros_like(actions), x, torch.tensor([0,1,0])) == 0


def test_landing_reserves_credit_until_descent_and_true_loading_without_farming():
    target = torch.ones(1, 2).long()*2; peak = torch.zeros(1, 2)
    active = torch.ones(1, 2).bool(); reset = torch.zeros(1).bool()
    total = 0.
    for velocity, force in [(1., 0.), (-.5, 0.), (0., 100.), (0., 100.), (-.5, 0.), (0., 100.)]:
        target, peak, credit = m.settlement_step(target, peak, target, torch.ones(1, 2),
            torch.zeros(1, 2), torch.full((1, 2), velocity), torch.full((1, 2), force), active, reset)
        if velocity > 0:
            assert credit.sum() == 0 and peak.sum() == 0
        total += float(credit.sum())
    assert total == pytest.approx(2.)
    target, peak, credit = m.settlement_step(target, peak, target+1, torch.ones(1, 2),
        torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2)*100, active, reset)
    assert credit.sum() == 2  # next target has its own finite budget


def test_load_biased_credit_requires_real_contact_for_most_of_the_score():
    target = torch.ones(1, 2).long()*2
    active = torch.ones(1, 2).bool(); reset = torch.zeros(1).bool()
    _, peak_air, credit_air = m.settlement_step(target, torch.zeros(1, 2), target,
        torch.ones(1, 2), torch.zeros(1, 2), torch.full((1, 2), -.5),
        torch.zeros(1, 2), active, reset, airborne_weight=.15, force_threshold=15.)
    _, peak_load, credit_load = m.settlement_step(target, torch.zeros(1, 2), target,
        torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2),
        torch.ones(1, 2)*100, active, reset, airborne_weight=.15, force_threshold=15.)
    assert credit_load.sum() > credit_air.sum() * 3
    assert peak_load.sum() > peak_air.sum()


@pytest.mark.parametrize('force', [0., 14., 15.])
def test_v2_contact_threshold_never_gives_loading_credit_before_load(force):
    target = torch.ones(1, 2).long()
    _, _, credit = m.settlement_step(target, torch.zeros(1, 2), target,
        torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2),
        torch.full((1, 2), force), torch.ones(1, 2).bool(), torch.zeros(1).bool(),
        airborne_weight=.15, force_threshold=15.)
    assert credit.sum() == 0


def test_v2_descent_and_loading_share_one_finite_tread_budget():
    target = torch.ones(1, 2).long()
    peak = torch.zeros(1, 2)
    credits = []
    for velocity, force in [(-.5, 0.), (0., 100.), (-.5, 0.), (0., 100.)]:
        target, peak, credit = m.settlement_step(target, peak, target,
            torch.ones(1, 2), torch.zeros(1, 2), torch.full((1, 2), velocity),
            torch.full((1, 2), force), torch.ones(1, 2).bool(), torch.zeros(1).bool(),
            airborne_weight=.15, force_threshold=15.)
        credits.append(float(credit.sum()))
    assert credits == pytest.approx([.3, 1.7, 0., 0.])


def test_positive_speed_credit_makes_progress_better_than_standing_without_speed_cap():
    speed = torch.tensor([-.7, 0., .28, .56, .7, 1.4, .56, .56])
    command = torch.tensor([.7]*6+[0., -.7])
    credit = m.ascent_speed_credit(speed, command, torch.ones(8).bool())
    torch.testing.assert_close(credit, torch.tensor([0., 0., .5, 1., 1., 1., 0., 0.]))
    assert m.ascent_speed_credit(speed, command, torch.zeros(8).bool()).count_nonzero() == 0


@pytest.mark.parametrize('fraction', [0., -1., float('nan'), float('inf'), 1.1])
def test_speed_credit_rejects_invalid_targets(fraction):
    with pytest.raises(ValueError):
        m.ascent_speed_credit(torch.ones(1), torch.ones(1), torch.ones(1).bool(), fraction)


@pytest.mark.parametrize('reset,enabled,alignment,height', [(True,True,1.,0.), (False,False,1.,0.),
    (False,True,0.,0.), (False,True,1.,.30)])
def test_landing_reset_closed_command_bypass_and_high_airborne(reset, enabled, alignment, height):
    target = torch.ones(1, 2).long()*2
    _, peak, credit = m.settlement_step(target, torch.zeros(1, 2), target, torch.full((1, 2), alignment),
        torch.full((1, 2), height), torch.zeros(1, 2), torch.ones(1, 2)*100,
        torch.ones(1, 2).bool()*enabled, torch.tensor([reset]))
    assert credit.sum() == 0  # high airborne is not descending or in contact plane


def test_closed_command_consumes_credit_without_later_catchup():
    target = torch.ones(1,2).long()
    def step(peak, active):
        return m.settlement_step(target, peak, target, torch.ones(1,2), torch.zeros(1,2),
            torch.zeros(1,2), torch.ones(1,2)*100, torch.ones(1,2).bool()*active, torch.zeros(1).bool())
    _, peak, credit = step(torch.zeros(1,2), False)
    assert credit.sum() == 0
    _, _, credit = step(peak, True)
    assert credit.sum() == 0


def test_deployment_quietness_never_crosses_case_or_phase_boundary():
    case = torch.tensor([0,0,0,0,1,1,1,1,1])
    phase = torch.tensor([1,1,1,2,1,1,1,2,2])
    assert m.coherent_triplets(case, phase).tolist() == [2,6]


def test_real_ppo_hook_gradients_without_training_or_parameter_updates():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    package_name = '_m20_targeted_cpu'
    package = types.ModuleType(package_name); package.__path__ = [str(SOURCE)]
    sys.modules[package_name] = package
    runner = load(package_name+'.v111_targeted_runner', SOURCE/'v111_targeted_runner.py')
    observations = TensorDict({'policy': torch.zeros(4,63), 'critic': torch.zeros(4,568)}, batch_size=[4])
    groups = {'actor':['policy'], 'critic':['critic']}
    actor = MLPModel(observations, groups, 'actor', 16, hidden_dims=[512,256,128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':1.,'std_type':'log'})
    critic = MLPModel(observations, groups, 'critic', 1, hidden_dims=[512,256,128])
    algorithm = runner.TargetedPPO(actor, critic, None, teacher_checkpoint=str(ROOT/'interview_prep/model_35300.pt'),
        replay_path=str(ROOT/'data/m20_recovery/reference_observations_v2.npz'), anchor_weight=2.,
        anchor_front_stair_weight=.04, anchor_batch_size=64, anchor_physical_ascent=True,
        anchor_ascent_wheel_differential_weight=.08, quiet_mean_weight=.02)
    algorithm.quiet_ramp_provider = lambda: 1.
    before = {n:p.detach().clone() for n,p in actor.named_parameters()}
    algorithm._anchor_step(algorithm.optimizer, (), {})
    assert 0 <= algorithm.symmetry_losses[-1] <= .05
    assert 0 <= algorithm.idle_losses[-1] <= .10
    assert 0 <= algorithm.quiet_losses[-1] <= .02
    assert 0 <= algorithm.workspace_losses[-1] <= .03
    gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert not algorithm.optimizer.state
    assert all(torch.equal(p, before[n]) for n,p in actor.named_parameters())
    assert all(not p.requires_grad for p in algorithm.teacher.parameters())


@pytest.fixture
def v2_algorithm():
    package_name = '_m20_targeted_v2_cpu'
    package = types.ModuleType(package_name); package.__path__ = [str(SOURCE)]
    sys.modules[package_name] = package
    v2 = load(package_name+'.v111_targeted_v2_runner', SOURCE/'v111_targeted_v2_runner.py')
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    observations = TensorDict({'policy': torch.zeros(4,63), 'critic': torch.zeros(4,568)}, batch_size=[4])
    actor = MLPModel(observations, {'actor':['policy']}, 'actor', 16, hidden_dims=[512,256,128],
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':1.,'std_type':'log'})
    critic = MLPModel(observations, {'critic':['critic']}, 'critic', 1, hidden_dims=[512,256,128])
    algorithm = v2.TargetedPPOV2(actor, critic, None,
        teacher_checkpoint=str(ROOT/'interview_prep/model_35300.pt'),
        replay_path=str(ROOT/'data/m20_recovery/reference_observations_v2.npz'), anchor_weight=2.,
        anchor_front_stair_weight=.04, anchor_batch_size=32, anchor_physical_ascent=True,
        anchor_ascent_wheel_differential_weight=.08, quiet_mean_weight=.02)
    return algorithm


def test_v2_runner_unfreezes_actor_and_connects_all_parameters_to_optimizer(v2_algorithm):
    algorithm = v2_algorithm
    assert all(p.requires_grad for p in algorithm.actor.parameters())
    optimizer_ids = {id(p) for group in algorithm.optimizer.param_groups for p in group['params']}
    assert all(id(p) in optimizer_ids for p in algorithm.actor.parameters())
    assert all(not p.requires_grad for p in algorithm.teacher.parameters())
    assert all(id(p) not in optimizer_ids for p in algorithm.teacher.parameters())


def test_v2_hook_backpropagates_to_early_actor_and_std_without_weight_update(v2_algorithm):
    algorithm = v2_algorithm
    assert algorithm.extra_anchor_loss().item() == 0
    with torch.no_grad():
        algorithm.actor.distribution.log_std_param[:] = -3
    before = {n: p.detach().clone() for n, p in algorithm.actor.named_parameters()}
    algorithm._anchor_step(algorithm.optimizer, (), {})
    assert algorithm.extra_losses[-1] > 0
    assert algorithm.actor.mlp[0].weight.grad.abs().sum() > 0
    assert algorithm.actor.mlp[2].weight.grad.abs().sum() > 0
    assert (algorithm.actor.distribution.log_std_param.grad < 0).all()
    assert all(p.grad is None for p in algorithm.teacher.parameters())
    assert all(torch.equal(p, before[n]) for n, p in algorithm.actor.named_parameters())
    assert not algorithm.optimizer.state


def test_v2_exploration_projection_preserves_learning_above_floor_and_releases_below(v2_algorithm):
    algorithm = v2_algorithm
    with torch.no_grad():
        algorithm.actor.distribution.log_std_param[::2] = -3
    before = algorithm.actor.distribution.log_std_param.detach().clone()
    algorithm._project_std_floor()
    after = algorithm.actor.distribution.log_std_param
    torch.testing.assert_close(after[::2], algorithm._log_std_floor()[::2])
    torch.testing.assert_close(after[1::2], before[1::2])
    assert after.requires_grad


def test_v2_reload_with_low_std_is_projected_before_next_rollout(v2_algorithm, monkeypatch):
    algorithm = v2_algorithm
    # Emulate a checkpoint load after construction, without sampling or PPO.
    with torch.no_grad():
        algorithm.actor.distribution.log_std_param[:] = -3
    def sample(self):
        torch.testing.assert_close(self.actor.distribution.log_std_param, self._log_std_floor())
        return 'sampled_after_floor'
    monkeypatch.setattr(type(algorithm).__mro__[1], 'act', sample)
    assert algorithm.act() == 'sampled_after_floor'


def test_v2_speed_scope_preserves_b1_deceleration_and_keeps_signal_after_skip(monkeypatch):
    from types import SimpleNamespace as NS
    name = '_m20_targeted_mdp_cpu'
    package = types.ModuleType(name)
    package.__path__ = [str(SOURCE/'tasks/manager_based/locomotion/velocity/mdp')]
    monkeypatch.setitem(sys.modules, name, package)
    if 'rl_training' not in sys.modules:
        parent = types.ModuleType('rl_training'); parent.__path__ = [str(SOURCE)]
        monkeypatch.setitem(sys.modules, 'rl_training', parent)
    monkeypatch.setitem(sys.modules, 'rl_training.targeted_reference', m)
    refine = types.ModuleType(name+'.v111_refine'); refine.state = lambda env: None
    monkeypatch.setitem(sys.modules, refine.__name__, refine)
    geometry = dict(ascent=torch.tensor([True, True, False, True, True, True]),
        flight_riser_count=torch.tensor([8]*6))
    geo = types.ModuleType(name+'.v135_terrain_geometry'); geo.v135_terrain_geometry = lambda env: geometry
    monkeypatch.setitem(sys.modules, geo.__name__, geo)
    mdp = load(name+'.v111_targeted', Path(package.__path__[0])/'v111_targeted.py')
    entry = dict(height_valid=torch.ones(6).bool(), entry_near=torch.tensor([False, True]+[False]*4),
        forward=torch.tensor([[1., 0.]]*6), anchor=torch.zeros(6, 2), tread=torch.full((6,), .3))
    data = NS(root_lin_vel_w=torch.tensor([[.56, 0., 0.]]*5+[[0., 0., 0.]]),
        root_pos_w=torch.tensor([[.6, 0., .5]]*4+[[4., 0., .5], [.6, 0., .5]]))
    command = torch.tensor([[.7, 0., 0.]]*3+[[0., 0., 0.]]+[[.7, 0., 0.]]*2)
    env = NS(_v111_first_tread=entry, _v111_adjacent=dict(ever_skipped=torch.ones(6, 2).bool()),
        scene={'robot': NS(data=data)}, command_manager=NS(get_command=lambda name: command))
    torch.testing.assert_close(mdp.ascent_speed_floor(env), torch.tensor([1., 0., 0., 0., 0., 0.]))
    command[0, 2] = .5
    assert not mdp.ascent_speed_floor(env).any(), 'explicit turning was rewarded for fixed-route drive'
    command[0, 2], command[0, 1] = 0., .5
    assert not mdp.ascent_speed_floor(env).any(), 'explicit lateral commands must keep their native objective'
    calls = []
    def landing(env, **options):
        calls.append(options)
        env._v111_targeted = {'total': torch.ones(6)}
    monkeypatch.setattr(mdp, '_landing_reward', landing)
    assert mdp.credit_metric_load_biased(env, slice(None)) == 1
    assert calls == [dict(airborne_weight=.15, force_threshold=15.)]


@pytest.mark.parametrize('version', ['TARGETED_TASK', 'TARGETED_V2_TASK', 'STEP_SUPPORT_TASK'])
def test_behavior_failure_continues_for_targeted_task(tmp_path, monkeypatch, version):
    fixture = load('targeted_guard_fixture', ROOT/'scripts/tools/test_m20_guarded_session.py')
    guard = fixture.guard
    runner, calls, baseline, logs, output = fixture.fixture(tmp_path, monkeypatch, fail_at=1)
    evaluate = guard.run_evaluation
    import json
    from types import SimpleNamespace
    def run(command, **kwargs):
        if 'm20_quiet_supplement.py' not in command[1]:
            return evaluate(command, **kwargs)
        path = Path(command[command.index('--output')+1]); path.mkdir()
        (path/'results.json').write_text('[]')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(guard, 'run_evaluation', run)
    guard.learn_guarded(runner, total_updates=101, segment_updates=100, baseline_path=baseline,
        output_path=output, log_dir=logs, task=getattr(guard, version))
    assert len(calls) == 2
    assert json.loads((output/'status.json').read_text())['retention_failures_seen']
