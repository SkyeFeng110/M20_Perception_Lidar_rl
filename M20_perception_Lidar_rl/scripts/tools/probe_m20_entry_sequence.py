"""Exercise the actual Isaac environment and gradients without any PPO update."""
import argparse
import json
import os
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--num_envs', type=int, default=64)
p.add_argument('--steps', type=int, default=350)
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
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_entry_transfer import state, flat_motion_state

task = 'M20-Stair-V11.1-EntrySequenceRepair-v0'
cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
assert cfg.v111_strict_entry_sequence
assert cfg.rewards.v111_entry_overshoot_speed is None
assert cfg.rewards.v111_entry_support is None and cfg.rewards.v111_entry_skip is None
assert cfg.rewards.v111_entry_simultaneous is None and cfg.rewards.v111_entry_pitch is None
assert cfg.rewards.v111_transition_action_impulse is None
assert cfg.rewards.joint_mirror.func.__name__ == 'after_rear_approach_reward'
assert cfg.rewards.world_forward_deficit.params['min_speed_fraction'] == .5
assert cfg.rewards.world_forward_deficit.func.__name__ == 'v111_world_forward_deficit_with_descent_support'
assert cfg.rewards.track_lin_vel_xy_exp.weight == 5.
assert cfg.commands.base_velocity.ranges.lin_vel_x == (-1., 1.)
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
flat_samples, eligible, credit, max_cost, full_speed = 0, 0, 0., 0., False
with torch.inference_mode():
    for _ in range(a.steps):
        obs, rewards, done, info = wrapped.step(policy(obs))
        s, flat = state(raw), flat_motion_state(raw)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(rewards).all()
        for key, cap in (('front_cost_used', 1.5), ('rear_cost_used', 1.),
                         ('posture_cost_used', .75), ('skip_cost_used', 1.)):
            assert (s['transfer'][key] <= cap+1.e-5).all()
        assert (flat['cost'] >= 0).all() and (flat['cost'] <= .500001).all()
        true_flat = flat['columns'][raw.scene.terrain.terrain_types]
        assert not flat['cost'][~true_flat].any() and not flat['quality'][~true_flat].any()
        flat_samples += int(flat['active'].sum())
        eligible = max(eligible, int(s['transfer']['eligible'].sum()))
        credit += float(s['credit'].sum())
        max_cost = max(max_cost, float(s['transfer']['cost_used'].max()))
        full_speed |= bool((raw.command_manager.get_command('base_velocity')[:, 0] == 1.).any())
    raw._reset_idx(torch.tensor([0, 1], device=raw.device))
    _, rewards, _, _ = wrapped.step(policy(wrapped.get_observations()))
    assert torch.isfinite(rewards).all() and not state(raw)['pending'].any()
runner.alg.optimizer.zero_grad()
runner.alg._anchor_step(runner.alg.optimizer, (), {})
grads = [p.grad for p in runner.alg.actor.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
assert not runner.alg.optimizer.state
assert all(torch.equal(v, before[k]) for k, v in runner.alg.actor.state_dict().items())
prediction = torch.randn(4, 16, device=raw.device, requires_grad=True)
zero = runner.alg.reference_loss(prediction, torch.zeros_like(prediction),
    torch.zeros(4, 63, device=raw.device), torch.full((4,), 2, device=raw.device),
    torch.ones(4, device=raw.device))
zero.backward()
assert zero == 0 and not prediction.grad.any()
assert flat_samples > 0 and eligible > 0 and full_speed
audit = {name: dict(weight=term.weight, func=term.func.__module__+':'+term.func.__name__,
                    params=repr(term.params))
         for name, term in vars(cfg.rewards).items() if term is not None and hasattr(term, 'func')}
result = dict(status='frozen_entry_sequence_runtime_passed', actor_input=63, actor_output=16,
    critic_input=568, envs=a.num_envs, steps=a.steps, maximum_entry_cost_used=max_cost,
    eligible_entry_envs=eligible, transfer_credit=credit, flat_samples=flat_samples,
    raw_full_speed_sampled=full_speed, post_b2_reference_samples=runner.alg.post_entry_samples,
    all_actor_parameters_trainable=True, no_b1_imitation=True, entry_slowdown_reward_disabled=True,
    native_velocity_rewards_active=True, flat_cost_on_stairs_zero=True,
    gradients_finite=True, optimizer_empty=True, model_weights_unchanged=True,
    no_training_started=True, reward_audit=audit)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
from isaaclab.utils.io import dump_yaml
dump_yaml(str(a.output.parent/'runtime_params/env.yaml'), cfg)
dump_yaml(str(a.output.parent/'runtime_params/agent.yaml'), agent)
print(json.dumps({key: value for key, value in result.items() if key != 'reward_audit'}), flush=True)
wrapped.close()
os._exit(0)
