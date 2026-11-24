"""Frozen Isaac wiring and gradient check; never learn/optimizer.step."""
import argparse
import json
import os
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--num_envs', type=int, default=40)
p.add_argument('--steps', type=int, default=220)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym
import torch
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.v111_recovery_runner import RecoveryRunner
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_entry_transfer import state

task = 'M20-Stair-V11.1-EntryTransfer-v0'
cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
old = parse_env_cfg('M20-Stair-V11.1-TractionTransition-v0', device=a.device, num_envs=a.num_envs)
for field in ('actions', 'observations', 'terminations'):
    assert repr(getattr(cfg, field).to_dict()) == repr(getattr(old, field).to_dict())
for field in ('track_lin_vel_xy_exp', 'world_forward_progress', 'world_forward_deficit'):
    assert repr(getattr(cfg.rewards, field)) == repr(getattr(old.rewards, field))
agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
env = gym.make(task, cfg=cfg)
raw = env.unwrapped
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = RecoveryRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(str(a.checkpoint.resolve()), load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=True, rnd=False))
before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
raw.common_step_counter = runner.current_learning_iteration*agent.num_steps_per_env
raw._m20_refine_start_step = raw.common_step_counter
wrapped.reset()
obs = wrapped.get_observations()
assert obs['policy'].shape == (a.num_envs, 63) and obs['critic'].shape == (a.num_envs, 568)
assert raw.action_manager.total_action_dim == 16
assert all(p.requires_grad for p in runner.alg.actor.parameters())
assert all(not p.requires_grad for p in runner.alg.teacher.parameters())
policy = runner.get_inference_policy(device=raw.device)
maximum_cost, credit, eligible = 0., 0., 0
with torch.inference_mode():
    for _ in range(a.steps):
        obs, rewards, done, info = wrapped.step(policy(obs))
        s = state(raw)
        assert torch.isfinite(rewards).all()
        for key in ('credit', 'cost', 'blocked', 'rocking'):
            assert torch.isfinite(s[key]).all() and (s[key] >= 0).all()
        assert (s['transfer']['cost_used'] <= 4.25001).all()
        maximum_cost = max(maximum_cost, float(s['transfer']['cost_used'].max()))
        credit += float(s['credit'].sum())
        eligible = max(eligible, int(s['transfer']['eligible'].sum()))
    raw._reset_idx(torch.tensor([0, 1], device=raw.device))
    obs, rewards, _, _ = wrapped.step(policy(wrapped.get_observations()))
    assert torch.isfinite(rewards).all() and not state(raw)['pending'].any()
# Autograd only: this invokes the actual auxiliary losses, but never Adam.
runner.alg.optimizer.zero_grad()
runner.alg._anchor_step(runner.alg.optimizer, (), {})
grads = [p.grad for p in runner.alg.actor.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert any(g.abs().sum() > 0 for g in grads)
assert not runner.alg.optimizer.state
assert all(torch.equal(v, before[k]) for k, v in runner.alg.actor.state_dict().items())
result = dict(status='frozen_entry_runtime_passed', actor_input=63, actor_output=16,
    critic_input=568, envs=a.num_envs, steps=a.steps, maximum_entry_cost_budget_used=maximum_cost,
    eligible_entry_envs=eligible, transfer_credit=credit,
    post_entry_reference_samples=runner.alg.post_entry_samples,
    all_actor_parameters_trainable=True, teacher_frozen=True,
    actual_anchor_gradients_finite=True, optimizer_empty=True, model_weights_unchanged=True,
    no_training_started=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
wrapped.close()
os._exit(0)
