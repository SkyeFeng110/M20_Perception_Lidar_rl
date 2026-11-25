"""Exercise installed Isaac joint processing without starting its simulation app."""
import ast
import importlib.util
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'source/rl_training/rl_training'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    contract = load('_m20_joint_contract_test', SOURCE / 'joint_target_limits.py')
    monkeypatch.setitem(sys.modules, 'rl_training.joint_target_limits', contract)
    # Run installed Isaac joint code verbatim; only its simulation-dependent
    # ActionTerm base is replaced by a small environment fixture.
    isaac = Path(importlib.util.find_spec('isaaclab').origin).parent
    string_utils = load('_m20_joint_string_test', isaac / 'utils/string.py')

    class ActionTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self._asset = env.scene[cfg.asset_name]
            self.num_envs, self.device = env.num_envs, env.device

    path = isaac / 'envs/mdp/actions/joint_actions.py'
    source = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [n for n in source.body if isinstance(n, ast.ClassDef)
              and n.name in ('JointAction', 'JointPositionAction')]
    module = types.ModuleType('_installed_m20_joint_actions')
    module.__dict__.update(torch=torch, ActionTerm=ActionTerm, string_utils=string_utils,
                           logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), module.__dict__)
    monkeypatch.setitem(sys.modules, 'isaaclab.envs.mdp.actions.joint_actions', module)
    limits = load('_m20_joint_action_test', SOURCE / 'tasks/manager_based/locomotion/velocity/mdp/v111_joint_limits.py')
    return contract, limits


def term_env(modules, n=2):
    contract, limits = modules
    default = torch.tensor([[0., -.3, .6, 0., -.3, .6, 0., .3, -.6, 0., .3, -.6]]).repeat(n, 1)
    asset = NS(num_joints=12, data=NS(default_joint_pos=default),
               find_joints=lambda names, preserve_order: (list(range(12)), list(names)))
    env = NS(num_envs=n, device='cpu', scene={'robot': asset})
    cfg = NS(asset_name='robot', joint_names=list(contract.LEG_JOINT_NAMES),
             preserve_order=True, scale={'.*_hipx_joint': .125, '^(?!.*_hipx_joint).*': .25},
             offset=0., clip=contract.joint_target_clip(), use_default_offset=True)
    term = limits.LimitedLegPositionAction(cfg, env)
    env.action_manager = NS(action=torch.zeros(n, 16), get_term=lambda name: term)
    return term, env


def test_bounds_scale_offset_history_and_ppo_samples_are_consistent(modules):
    contract, limits = modules
    term, env = term_env(modules)
    sampled = torch.zeros(2, 12)
    sampled[0] = torch.tensor([1., -1., 1.] * 4) * 100
    sampled[1] = torch.linspace(-1., 1., 12)
    original = sampled.clone()
    term.process_actions(sampled)
    bounds = torch.tensor(contract.joint_target_bounds())
    requested = sampled * term._scale + term._offset
    expected = torch.clamp(requested, bounds[:, 0], bounds[:, 1])
    torch.testing.assert_close(term.processed_actions, expected)
    torch.testing.assert_close(term.raw_actions, original)
    torch.testing.assert_close(sampled, original)
    torch.testing.assert_close(term.executed_actions[1], original[1], rtol=0, atol=0)
    torch.testing.assert_close(term.executed_actions*term._scale+term._offset, expected)
    env.action_manager.action[:, :12] = sampled
    env.action_manager.action[:, 12:] = 3.
    env._m20_last_action_command = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    history = limits.executed_last_action(env)
    torch.testing.assert_close(history[:, :12], term.executed_actions)
    assert not history[0, 12:].any() and (history[1, 12:] == 3).all()
    torch.testing.assert_close(env.action_manager.action[:, :12], original)
    assert limits.joint_target_violation_fraction(env, torch.arange(2)) == .5
    assert limits.joint_target_excess_cost(env).tolist() == [5., 0.]
    term.reset(torch.tensor([0]))
    assert not term.frames[0] and term.frames[1] == 1
    assert not term.executed_actions[0].any()
    term.reset()
    assert not term.frames.any()


def test_bad_action_fails_before_changing_target_and_asymmetric_bounds_are_preserved(modules):
    contract, _ = modules
    term, _ = term_env(modules)
    old = term.processed_actions.clone()
    with pytest.raises(ValueError, match='Nonfinite'):
        term.process_actions(torch.full((2, 12), float('nan')))
    torch.testing.assert_close(term.processed_actions, old)
    bounds = np.asarray(contract.joint_target_bounds())
    assert bounds[0, 0] == -.406 and bounds[3, 0] == -.581
    assert bounds[7, 0] == -2.533 and bounds[1, 0] == -2.236


def test_same_extreme_actions_produce_same_isaac_and_mujoco_targets(modules, monkeypatch):
    contract, limits = modules
    term, env = term_env(modules)
    monkeypatch.syspath_prepend(str(SOURCE.parents[3] / 'mujoco_validation/deploy/mujoco_v135'))
    from policy import SavedPolicy
    policy = object.__new__(SavedPolicy)
    policy.nominal = np.r_[term._offset[0].numpy(), np.zeros(4)]
    policy.action_scale = np.r_[term._scale[0].numpy(), np.full(4, 20.)]
    policy.leg_target_limits = np.asarray(contract.joint_target_bounds())
    policy.limited_leg_action = True
    policy.metadata = {'action_history_function': 'executed_last_action'}
    sampled = torch.randn(2, 16, generator=torch.Generator().manual_seed(47))*30
    term.process_actions(sampled[:, :12])
    env.action_manager.action.copy_(sampled)
    expected_history = limits.executed_last_action(env).numpy()
    for i, action in enumerate(sampled.numpy()):
        np.testing.assert_allclose(policy.position_target(action)[:12], term.processed_actions[i].numpy(), atol=2e-7)
        np.testing.assert_allclose(policy.history_action(action), expected_history[i], atol=2e-6)
