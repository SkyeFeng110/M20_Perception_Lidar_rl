"""Reward-only refinement, with finite landing credit and local quietness proxies.

No actor inputs, executed actions, actuator gains or base drive rewards change.
Pure tensor functions below are also exercised without starting a simulator.
"""
import math
import torch


def landing_credit(target, peak, expected, score, active):
    """At most one unit per physical target; holding/backtracking cannot farm it.

    A newly observed target starts at its current potential, including after a
    skip/reset. Inactive samples update the ledger without paying catch-up.
    """
    score = score.clamp(0, 1)
    prior = torch.where(target == expected, peak, score)
    return expected.clone(), torch.maximum(prior, score), (score-prior).clamp_min(0)*active


def compact_quality(command, velocity, yaw_rate, workspace, blend, cfg):
    straight = (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03)
    progress = (velocity[:, 0]*command[:, 0].sign()/command[:, 0].abs().clamp_min(.1)).clamp(0, 1)
    tracking = torch.exp(-((velocity[:, 0]-command[:, 0])/.5).square())
    rate_error = (yaw_rate/.08).square() + (velocity[:, 1]/.12).square()
    # Independent rational terms keep posture feedback usable even when yaw is
    # poor, and vice versa. The original product exp(-shape-yaw) saturated.
    quality = .6/(1+workspace) + .4/(1+rate_error)
    return progress*tracking*quality*blend*straight*(command[:, 0].abs() > .1)


def chatter_score(delta, previous_delta, stair, deadband=.02):
    """Penalize rapid reversals, not steady drive or a monotonic leg lift."""
    reversal = (delta*previous_delta < 0)
    magnitude = (torch.minimum(delta.abs(), previous_delta.abs())-deadband).clamp_min(0)
    scale = torch.tensor([.25]*12+[.5]*4, device=delta.device, dtype=delta.dtype)
    energy = (magnitude/scale).square()
    score = energy/(1+energy)*reversal
    # Keep necessary sagittal stair motions inexpensive; no amplitude cost.
    weights = torch.ones_like(score)
    weights[:, [1, 2, 4, 5, 7, 8, 10, 11]] = torch.where(stair[:, None], .2, 1.)
    return (score*weights).mean(1)


def contact_step(s, force, normal_velocity, reset, dt, *, on=15., off=5., min_air=.06):
    """Hysteretic wheel touchdown and short load-build window, at policy rate.

    Vertical forces/velocities approximate horizontal tread contact in Isaac.
    They are not a calibrated sound model or a full collision-normal sensor.
    """
    s['contact'][reset] = force[reset] >= on
    s['air'][reset] = 0
    s['window'][reset] = 0
    s['force'][reset] = force[reset]
    s['velocity'][reset] = normal_velocity[reset]
    contact = torch.where(s['contact'], force > off, force >= on)
    touch = contact & ~s['contact'] & (s['air'] >= min_air) & ~reset[:, None]
    approach = torch.maximum(-s['velocity'], -normal_velocity).clamp_min(0)
    impact = (approach/.75).square()
    impact = impact/(1+impact)*touch
    window = torch.where(touch, .06, (s['window']-dt).clamp_min(0))
    load = ((force-s['force']).clamp_min(0)/200.).square()
    load = load/(1+load)*(window > 0)*contact
    s.update(contact=contact, air=torch.where(contact, 0., s['air']+dt),
             window=window, force=force.clone(), velocity=normal_velocity.clone())
    return (.7*impact+.3*load).mean(1), touch.float().sum(1)


def noise_ramp(stage_steps, rollout=96, warmup=200, full=1200):
    return max(0., min(1., (stage_steps/rollout-warmup)/(full-warmup)))


def _new_state(env):
    z = lambda *shape: torch.zeros((env.num_envs, *shape), device=env.device)
    return dict(step=-1, pending=z().bool(), length=z().long(), target=z(2).long(), peak=z(2),
                delta=z(16), contact=z(4).bool(), air=z(4), window=z(4),
                force=z(4), velocity=z(4), top_frames=z().long(), arrived=z().bool(),
                samples=z(), contact_sum=z(), chatter_sum=z(), touch_sum=z(),
                support_credit_sum=z(), route_sum=z(), yaw_max=z(), reach_max=z())


def state(env):
    from .v11_continuous import v11_state
    from .v111_entry_flat import flat_state
    from .v135_terrain_geometry import v135_terrain_geometry
    v11_state(env)
    if not hasattr(env, '_v111_refine'):
        env._v111_refine = _new_state(env)
    s = env._v111_refine
    step = int(env.common_step_counter)
    if s['step'] == step:
        return s
    reset = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    if s['step'] == -1:
        reset[:] = True
    for key in ('target', 'peak', 'delta', 'top_frames', 'arrived', 'samples',
                'contact_sum', 'chatter_sum', 'touch_sum', 'support_credit_sum',
                'route_sum', 'yaw_max', 'reach_max'):
        s[key][reset] = 0
    robot = env.scene['robot']
    cfg = env.cfg.v111_refine
    entry, adjacent = env._v111_first_tread, env._v111_adjacent
    geometry = v135_terrain_geometry(env)
    wheel_ids = env._v111_recovery_wheels
    wheels = robot.data.body_pos_w[:, wheel_ids]
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts, 2].clamp_min(0)
    # For horizontal treads r=(0,0,-R), the rotational contribution to normal
    # velocity is zero. Never charge forward hub speed as wheel slip/noise.
    normal_velocity = robot.data.body_link_lin_vel_w[:, wheel_ids, 2]
    s['contact_cost'], touches = contact_step(s, force, normal_velocity, reset, float(env.step_dt))
    delta = env.action_manager.action-env.action_manager.prev_action
    s['chatter_cost'] = chatter_score(delta, s['delta'], geometry['ascent'] | v11_state(env)['descent']['target_active'])
    s['chatter_cost'][reset] = 0
    s['delta'] = delta.clone()

    expected = adjacent['overshoot_target']
    x = ((wheels[:, :2, :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    bottom = wheels[:, :2, 2]-env.cfg.v11_perception.wheel_radius
    margin = torch.minimum(torch.full_like(entry['tread'], .025), entry['tread']*.1)[:, None]
    start = (expected-1)*entry['tread'][:, None]+margin
    end = expected*entry['tread'][:, None]-margin
    outside = torch.maximum(start-x, x-end).clamp_min(0)
    plane = entry['lower'][:, None]+expected*entry['height'][:, None]
    left = torch.stack((-entry['forward'][:, 1], entry['forward'][:, 0]), 1)
    lateral = ((wheels[:, :2, :2]-env.scene.env_origins[:, None, :2])*left[:, None]).sum(-1)
    width = (left.abs()*geometry['radii']).sum(-1)[:, None] + (expected-1)*entry['tread'][:, None]
    in_course = lateral.abs() <= width-env.cfg.v11_perception.wheel_radius-.02
    score = torch.exp(-(outside/.15).square()-((bottom-plane)/.16).square())
    score *= in_course & (x <= end) & (bottom >= plane-entry['height'][:, None]+.04)
    command = env.command_manager.get_command('base_velocity')
    active = ((geometry['ascent'] & entry['height_valid'] & ~reset & (command[:, 0] > .1))[:, None]
              & entry['entry_approached'] & (expected >= 2) & (expected <= geometry['flight_riser_count'][:, None]))
    s['target'], s['peak'], credit = landing_credit(s['target'], s['peak'], expected, score, active)
    s['landing_credit'] = credit.sum(1)

    # Physical arrival remains attainable after a skip, while the original
    # stricter whole-flight reward and acceptance still require every tread.
    count = geometry['flight_riser_count']
    top = entry['lower']+count*entry['height']
    wheel_x = ((wheels[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    supported = (force >= 15.) & ((wheels[..., 2]-env.cfg.v11_perception.wheel_radius-top[:, None]).abs() <= .035)
    candidate = (geometry['ascent'] & entry['height_valid'] & supported.all(1)
        & (wheel_x >= (count-1)[:, None]*entry['tread'][:, None]).all(1)
        & (-robot.data.projected_gravity_b[:, 2] >= .8) & (robot.data.root_pos_w[:, 2]-top >= .30))
    s['top_frames'] = torch.where(candidate & ~reset, s['top_frames']+1, 0)
    ready = s['top_frames'] >= math.ceil(.06/float(env.step_dt))
    s['arrival'] = ready & ~s['arrived']
    s['arrived'] |= ready

    term = env.command_manager.get_term('base_velocity')
    yaw = robot.data.heading_w-term.refine_heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    offset = robot.data.root_pos_w[:, :2]-term.refine_origin
    lateral = -offset[:, 0]*term.refine_heading.sin()+offset[:, 1]*term.refine_heading.cos()
    straight = (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03) & ~term.is_heading_env
    route_error = (yaw/cfg.route_yaw_scale).square()+(lateral/cfg.route_lateral_scale).square()
    s['route_cost'] = route_error/(1+route_error)*straight
    flat = flat_state(env)
    s['quality'] = compact_quality(command, robot.data.root_lin_vel_b, robot.data.root_ang_vel_b[:, 2],
                                   flat['workspace_cost'], flat['blend'], cfg)
    s['ramp'] = noise_ramp(step-getattr(env, '_m20_refine_start_step', step))
    s['samples'] += 1
    s['contact_sum'] += s['contact_cost']
    s['chatter_sum'] += s['chatter_cost']
    s['touch_sum'] += touches
    s['support_credit_sum'] += s['landing_credit']
    s['route_sum'] += s['route_cost']
    s['yaw_max'] = torch.maximum(s['yaw_max'], yaw.abs()*straight)
    s['reach_max'] = torch.maximum(s['reach_max'], flat['reach_max']*flat['active'])
    s['length'] = env.episode_length_buf.clone()
    s['pending'][:] = False
    s['step'] = step
    return s


def quality_reward(env):
    return state(env)['quality']


def route_cost(env):
    return -state(env)['route_cost']


def landing_reward(env):
    return state(env)['landing_credit']/float(env.step_dt)


def arrival_reward(env):
    return state(env)['arrival'].float()/float(env.step_dt)


def contact_cost(env):
    s = state(env)
    return -s['ramp']*s['contact_cost']


def metric(env, env_ids, key):
    s = state(env)
    if key == 'noise_ramp':
        return s['ramp']
    value = s[key]
    if key in ('contact_sum', 'chatter_sum', 'route_sum'):
        value = value/s['samples'].clamp_min(1)
    return value[env_ids].float().mean()
