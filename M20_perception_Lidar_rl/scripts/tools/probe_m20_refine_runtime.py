"""Frozen Isaac rollout of RecoveryRefine; never calls learn or optimizer.step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--task', choices=['M20-Stair-V11.1-RecoveryRefine-v0',
    'M20-Stair-V11.1-TargetedCorrection-v0',
    'M20-Stair-V11.1-TargetedCorrectionV2-v0',
    'M20-Stair-V11.1-StepSupport-v0',
    'M20-Stair-V11.1-StepRetention-v0', 'M20-Stair-V11.1-TractionTransition-v0'], default='M20-Stair-V11.1-RecoveryRefine-v0')
p.add_argument('--num_envs', type=int, default=40)
p.add_argument('--steps', type=int, default=180)
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
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_refine as refine
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_targeted as targeted
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_step_support as support
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import v135_terrain_geometry
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import column_names, ascent_name

TASK = a.task
cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
old = parse_env_cfg('M20-Stair-V11.1-AscentRecovery-v0', device=a.device, num_envs=a.num_envs)
for section in ('actions', 'observations', 'terminations'):
    assert repr(getattr(cfg, section).to_dict()) == repr(getattr(old, section).to_dict())
for reward in ('track_lin_vel_xy_exp', 'world_forward_progress', 'world_forward_deficit'):
    assert repr(getattr(cfg.rewards, reward)) == repr(getattr(old.rewards, reward))
if TASK.endswith(('StepSupport-v0', 'StepRetention-v0', 'TractionTransition-v0')):
    terrain = cfg.scene.terrain.terrain_generator
    columns = column_names(terrain)
    old_columns = column_names(old.scene.terrain.terrain_generator)
    assert len(columns) == len(old_columns) == 20
    assert all(a == b for a, b in zip(columns, old_columns) if not ascent_name(a))
    focused = [terrain.sub_terrains[n] for n in columns if ascent_name(n)
               and terrain.sub_terrains[n].step_height_range == (.16, .16)]
    assert len(focused) == 4
    assert sorted(t.step_width for t in focused) == [.26, .26, .30, .30]
cfg.seed = 42
agent = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
assert agent.algorithm.anchor_physical_ascent
env = gym.make(TASK, cfg=cfg)
raw = env.unwrapped
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = RecoveryRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(str(a.checkpoint.resolve()), load_cfg={'actor': True, 'critic': True, 'optimizer': False,
                                               'iteration': True, 'rnd': False})
weights = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
raw.common_step_counter = runner.current_learning_iteration*agent.num_steps_per_env
raw._m20_refine_start_step = raw.common_step_counter
wrapped.reset()
observations = wrapped.get_observations()
assert observations['policy'].shape == (a.num_envs, 63)
assert observations['critic'].shape == (a.num_envs, 568)
assert raw.action_manager.total_action_dim == 16
term = raw.command_manager.get_term('base_velocity')
if TASK.endswith('TractionTransition-v0'):
    selected = term.transition_active.nonzero(as_tuple=True)[0]
    assert len(selected) >= 6
    # Exercise all six profiles deterministically in this frozen wiring probe.
    term.transition_profile[selected] = torch.arange(len(selected), device=raw.device) % 6
policy = runner.get_inference_policy(device=raw.device)
max_noise_cost, contacts, landing_credit = 0., 0, 0.
max_ascent_speed_credit = 0.
support_credit = 0.
with torch.inference_mode():
    for i in range(a.steps):
        observations, rewards, done, info = wrapped.step(policy(observations))
        assert torch.isfinite(rewards).all()
        s = refine.state(raw)
        if hasattr(raw, '_v111_targeted'):
            assert torch.isfinite(raw._v111_targeted['credit']).all()
            assert (raw._v111_targeted['credit'] >= 0).all()
        assert all(torch.isfinite(s[key]).all() for key in ('contact_cost', 'chatter_cost', 'landing_credit', 'route_cost'))
        max_noise_cost = max(max_noise_cost, float((.15*s['contact_cost']).max()))
        contacts += int(s['touch_sum'].sum()) if i == a.steps-1 else 0
        actual_credit = raw._v111_targeted['credit'] if hasattr(raw, '_v111_targeted') else s['landing_credit']
        landing_credit += float(actual_credit.sum())
        if TASK.endswith(('TargetedCorrectionV2-v0', 'StepSupport-v0', 'StepRetention-v0', 'TractionTransition-v0')):
            speed_credit = targeted.ascent_speed_floor(raw)
            assert torch.isfinite(speed_credit).all() and (speed_credit >= 0).all()
            assert (speed_credit <= 1).all()
            inactive = ~v135_terrain_geometry(raw)['ascent'] | raw._v111_first_tread['entry_near']
            command = raw.command_manager.get_command('base_velocity')
            inactive |= (command[:, 1].abs() >= .05) | (command[:, 2].abs() >= .03)
            assert not speed_credit[inactive].any()
            max_ascent_speed_credit = max(max_ascent_speed_credit, float(speed_credit.max()))
        if TASK.endswith(('StepSupport-v0', 'StepRetention-v0', 'TractionTransition-v0')):
            support_state = support.state(raw)
            assert torch.isfinite(support_state['credit']).all()
            assert (support_state['credit'] >= 0).all() and (support_state['peak'] <= 1).all()
            assert not support_state['credit'][~v135_terrain_geometry(raw)['ascent']].any()
            support_credit += float(support_state['credit'].sum())
        if TASK.endswith(('StepRetention-v0', 'TractionTransition-v0')):
            from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_step_retention import route_reward
            route = route_reward(raw)
            assert torch.isfinite(route).all() and (route >= 0).all() and (route <= 1.).all()
            assert not route[~v135_terrain_geometry(raw)['ascent']].any()
        if TASK.endswith('TractionTransition-v0'):
            from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_traction_transition as traction
            traction_state = traction.state(raw)
            for key in ('quality', 'cost', 'pitch_cost', 'flat_quality', 'transition_impulse',
                        'entry_pitch', 'entry_simultaneous', 'entry_lead'):
                assert torch.isfinite(traction_state[key]).all()
                assert (traction_state[key] >= 0).all()
            assert (traction_state['quality'] <= 1).all() and (traction_state['cost'] <= 1).all()
            assert (traction_state['entry_simultaneous'] <= 1).all()
            assert (traction_state['entry_lead'] <= 1).all()
            assert traction_state['supported'].shape == (a.num_envs, 4)
            assert not traction_state['quality'][~traction_state['straight']].any()
            assert cfg.rewards.v111_entry_simultaneous.weight == .80
            assert cfg.rewards.v111_transition_action_impulse.weight == .35
            assert cfg.rewards.v111_entry_lead.weight == .10
            from rl_training.traction_transition import transition_speed
            selected = term.transition_active
            expected = transition_speed((term.refine_elapsed[selected]-raw.step_dt).clamp_min(0),
                                        term.transition_profile[selected])
            torch.testing.assert_close(term.command[selected, 0], expected)
        if i < 95:
            assert (term.command[term.refine_cohort] == 0).all()
        if i == 135:
            assert (term.command[term.refine_cohort, 0] >= .5).any()
    # Test the noise curriculum without doing 1200 PPO updates.
    raw._m20_refine_start_step = raw.common_step_counter-1200*96
    observations, _, _, _ = wrapped.step(policy(observations))
    assert refine.state(raw)['ramp'] == 1.
    # Exact reset marks matter even if an episode lasted a single step.
    raw._reset_idx(torch.tensor([0, 1], device=raw.device))
    observations = wrapped.get_observations()
    observations, rewards, _, _ = wrapped.step(policy(observations))
    assert torch.isfinite(rewards).all()
    assert not refine.state(raw)['pending'].any()
assert not runner.alg.optimizer.state
assert all(torch.equal(v, weights[k]) for k, v in runner.alg.actor.state_dict().items())
assert max_noise_cost <= .15
if TASK.endswith('TargetedCorrection-v0'):
    assert runner.alg.__class__.__name__ == 'TargetedPPO'
    assert runner.alg.symmetry_weight == .05 and runner.alg.idle_mean_weight == .10
    assert cfg.rewards.v111_adjacent_support.weight == 1.6
    assert not raw._v111_targeted['pending'].any()
if TASK.endswith('TargetedCorrectionV2-v0'):
    assert runner.alg.__class__.__name__ == 'TargetedPPOV2'
    assert cfg.rewards.v111_adjacent_support.weight == 1.6
    assert cfg.rewards.v111_refine_landing.func.__name__ == 'landing_reward_load_biased'
    assert cfg.curriculum.v111_refine_support_credit_sum.func.__name__ == 'credit_metric_load_biased'
    assert all(p.requires_grad for p in runner.alg.actor.parameters())
    assert all(not p.requires_grad for p in runner.alg.teacher.parameters())
    assert torch.all(runner.alg.actor.distribution.log_std_param.exp() >=
                     torch.tensor([.18] * 12 + [.35] * 4, device=raw.device))
    assert max_ascent_speed_credit > 0
if TASK.endswith(('StepSupport-v0', 'StepRetention-v0', 'TractionTransition-v0')):
    assert runner.alg.__class__.__name__ == ('TractionTransitionPPO' if TASK.endswith('TractionTransition-v0') else 'StepRetentionPPO' if TASK.endswith('StepRetention-v0') else 'TargetedPPOV2')
    assert all(p.requires_grad for p in runner.alg.actor.parameters())
    assert all(not p.requires_grad for p in runner.alg.teacher.parameters())
    assert cfg.rewards.v111_refine_landing.func is support.landing_reward
    assert cfg.rewards.v111_adjacent_support.weight == 2.4
    assert runner.alg.lam == .97
    assert support_credit > 0
    geometry = v135_terrain_geometry(raw)
    focus = geometry['ascent'] & ((geometry['step_width'] == .26) | (geometry['step_width'] == .30))
    assert focus.any()
    chosen = torch.tensor([.5, .7, 1.], device=raw.device)
    # The takeover cohort's stored target uses the same benchmark speeds;
    # its current command may still be idle or partway through the ramp.
    assert torch.isclose(term.refine_speed[focus & term.refine_cohort, None], chosen).any(1).all()
    assert not support.state(raw)['pending'].any()
result = dict(status='frozen_runtime_passed', task=TASK, envs=a.num_envs, steps=a.steps,
              checkpoint=str(a.checkpoint.resolve()),
              actor_input=63, actor_output=16, critic_input=568, maximum_noise_rate_cost=max_noise_cost,
              touchdown_events=contacts, landing_credit=landing_credit, no_training_started=True,
              model_weights_unchanged=True)
if TASK.endswith(('StepSupport-v0', 'StepRetention-v0', 'TractionTransition-v0')):
    result.update(all_actor_parameters_trainable=True, teacher_parameters_frozen=True,
                  support_credit=support_credit, gae_lambda=runner.alg.lam,
                  no_credit_on_non_ascent=True, focused_terrain_columns=4,
                  focused_step_height=.16, focused_treads=[.26, .30],
                  focused_speeds=[.5, .7, 1.])
if TASK.endswith(('StepRetention-v0', 'TractionTransition-v0')):
    result.update(successful_reference_cases=len(runner.alg.success_members),
                  reference_sha256=runner.alg.reference_sha256,
                  all_actor_parameters_trainable=all(p.requires_grad for p in runner.alg.actor.parameters()),
                  forward_route_weight=cfg.rewards.v111_retention_forward_route.weight,
                  learning_rate=runner.alg.learning_rate)
if TASK.endswith('TractionTransition-v0'):
    training_root = Path(__file__).resolve().parents[2]
    paths = [
        'source/rl_training/rl_training/traction_transition.py',
        'source/rl_training/rl_training/v111_traction_transition_runner.py',
        'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v111_traction_transition.py',
        'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v111_traction_transition_env_cfg.py',
    ]
    result.update(traction_contact_samples=int(traction_state['supported'].sum()),
                  privileged_horizontal_contact_only=True,
                  no_policy_input_changes=True, fresh_optimizer_required=True,
                  sampled_transition_environments=int(term.transition_active.sum()),
                  six_transition_profiles_exercised=True,
                  source_sha256={p: hashlib.sha256((training_root/p).read_bytes()).hexdigest() for p in paths})
if TASK.endswith('TargetedCorrectionV2-v0'):
    result.update(all_actor_parameters_trainable=True, teacher_parameters_frozen=True,
        maximum_ascent_speed_credit=max_ascent_speed_credit,
        speed_floor_excludes_b1_and_non_ascent=True, support_credit_metric_matches_reward=True,
        exploration_floor_leg=runner.alg.std_floor_leg, exploration_floor_wheel=runner.alg.std_floor_wheel)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print('[REFINE RUNTIME] '+json.dumps(result), flush=True)
wrapped.close()
# The headless Kit shutdown can wait indefinitely after native env teardown
# on this host. Match the training worker's successful process-exit path.
os._exit(0)
