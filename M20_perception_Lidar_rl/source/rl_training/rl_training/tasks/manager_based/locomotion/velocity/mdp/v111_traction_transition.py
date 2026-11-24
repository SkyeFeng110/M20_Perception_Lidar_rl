"""Privileged horizontal-contact traction and flat acceleration refinement."""
import torch
from rl_training.traction_transition import (
    transition_speed, horizontal_patch, rolling_residual, traction_terms, flat_pitch_cost,
    transition_action_cost, entry_order_step)
from .v111_step_support import StepSupportVelocityCommand


class TractionTransitionVelocityCommand(StepSupportVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.transition_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.transition_profile = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        from .v135_terrain_geometry import column_names, ascent_name
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        names = column_names(self._env.cfg.scene.terrain.terrain_generator)
        eligible = torch.tensor([n == 'flat' or ascent_name(n) for n in names], device=self.device)
        # Half of the eligible flat/ascent environments receive an explicit
        # command transition. The other half retain the inherited distribution
        # so normal steering, descent and recovery are not overwritten.
        selected = eligible[self._env.scene.terrain.terrain_types[ids]] & ((ids % 8 >= 1) & (ids % 8 <= 4))
        self.transition_active[ids] = selected
        chosen = ids[selected]
        self.transition_profile[chosen] = torch.randint(6, (len(chosen),), device=self.device)
        self.vel_command_b[chosen] = 0
        self.is_heading_env[chosen] = False
        self.is_standing_env[chosen] = False
        self.v11_path_heading_w[chosen] = self.refine_heading[chosen]
        self.v11_path_origin_w[chosen] = self.refine_origin[chosen]

    def _update_command(self):
        super()._update_command()
        ids = self.transition_active
        # Parent advances the elapsed clock once. Use the same tick here.
        self.vel_command_b[ids, 0] = transition_speed(
            (self.refine_elapsed[ids]-float(self._env.step_dt)).clamp_min(0), self.transition_profile[ids])
        self.vel_command_b[ids, 1:] = 0


def state(env):
    from isaaclab.utils.warp import raycast_mesh
    from .v111_step_support import state as support_state
    from .v111_entry_flat import flat_state
    from .v135_terrain_geometry import v135_terrain_geometry
    support_state(env)
    step = int(env.common_step_counter)
    if hasattr(env, '_v111_traction') and env._v111_traction['step'] == step:
        return env._v111_traction
    robot = env.scene['robot']
    wheels = env._v111_recovery_wheels
    pos = robot.data.body_link_pos_w[:, wheels]
    linear = robot.data.body_link_lin_vel_w[:, wheels]
    angular = robot.data.body_link_ang_vel_w[:, wheels]
    scanner = env.scene.sensors['v10_pgtt_scanner']
    # Uses the scanner's already built static terrain mesh, but clean rays at
    # each wheel are reward-only: no new actor/critic observations or sensors.
    offsets = pos.new_tensor([[0, 0, .3], [.03, 0, .3], [-.03, 0, .3],
                              [0, .03, .3], [0, -.03, .3]])
    starts = pos[:, :, None]+offsets
    directions = torch.zeros_like(starts); directions[..., 2] = -1
    hits, _, normals, _ = raycast_mesh(starts, directions,
        scanner.meshes[scanner.cfg.mesh_prim_paths[0]], max_dist=1.5, return_normal=True)
    radius = float(env.cfg.v11_perception.wheel_radius)
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts, 2]
    supported = horizontal_patch(pos[..., 2]-radius, hits, normals, force, linear[..., 2])
    residual = rolling_residual(linear, angular, radius)
    term = env.command_manager.get_term('base_velocity')
    command = env.command_manager.get_command('base_velocity')
    # Flat native commands are relative to their resample heading; the
    # ancestor's stair path cache is not initialized for every flat sample.
    heading = torch.where(v135_terrain_geometry(env)['ascent'],
                          term.v11_path_heading_w, term.refine_heading)
    speed = (robot.data.root_lin_vel_w[:, 0]*heading.cos()
             +robot.data.root_lin_vel_w[:, 1]*heading.sin())
    straight = ((command[:, 1].abs() < .05) & (command[:, 2].abs() < .03) & ~term.is_heading_env)
    quality, cost, slip = traction_terms(residual, supported, speed, command[:, 0])
    flat = flat_state(env)
    pitch_cost_value = flat_pitch_cost(robot.data.projected_gravity_b, robot.data.root_ang_vel_b[:, 1])
    progress = (speed/command[:, 0].clamp_min(.1)).clamp(0, 1)*(command[:, 0] > .1)
    entry = env._v111_first_tread
    entry_near = entry.get('entry_near', torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    entry_active = (v135_terrain_geometry(env)['ascent'] & entry.get('height_valid', torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
                    & ~entry.get('done', torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
                    & ~entry.get('entry_failed', torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)))
    action = env.action_manager.action
    previous_action = env.action_manager.prev_action
    previous_command = getattr(env, '_m20_traction_previous_command', torch.zeros_like(command[:, 0]))
    impulse = transition_action_cost(action, previous_action, command[:, 0], previous_command,
                                    straight=straight, entry_near=entry_near)
    pitch = torch.atan2(robot.data.projected_gravity_b[:, 0],
                        torch.linalg.vector_norm(robot.data.projected_gravity_b[:, 1:], dim=1))
    approach_pitch = ((pitch.abs()-.08).clamp_min(0)/.14).square().clamp_max(2.)
    approach_rate = ((robot.data.root_ang_vel_b[:, 1].abs()-.45).clamp_min(0)/.8).square().clamp_max(2.)
    entry_pitch = (0.7*approach_pitch+0.3*approach_rate)*entry_near*straight
    support = entry.get('entry_support_steps', torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long)) > 0
    bottom = robot.data.body_pos_w[:, env._v111_recovery_wheels[:2], 2]-radius
    air = bottom > entry.get('lower', torch.zeros(env.num_envs, device=env.device))[:, None]+float(env.cfg.v111_entry_flat.entry_lift_min)
    if not hasattr(env, '_v111_traction_order'):
        env._v111_traction_order = dict(previous_support=torch.zeros_like(support), previous_air=torch.zeros_like(air),
            age=torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
            first_side=torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            first_kind=torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            latched=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool), last=env.episode_length_buf.clone())
    order = env._v111_traction_order
    reset = env.episode_length_buf < order['last']
    for key in ('previous_support', 'previous_air', 'age', 'first_side', 'first_kind', 'latched'):
        order[key][reset] = {'previous_support': False, 'previous_air': False, 'age': 0,
                             'first_side': -1, 'first_kind': -1, 'latched': False}[key]
    order.update(entry_order_step(order['previous_support'], support, order['previous_air'], air,
                                  entry_active, order['age'], order['first_side'], order['first_kind'],
                                  order['latched']))
    order['last'] = env.episode_length_buf.clone()
    env._m20_traction_previous_command = command[:, 0].detach().clone()
    env._v111_traction = dict(step=step, quality=quality*straight, cost=cost*straight,
        pitch_cost=pitch_cost_value*flat['blend']*straight*(command[:, 0] > .1),
        flat_quality=progress*(1-pitch_cost_value).clamp_min(0)*flat['blend']*straight/(1+flat['workspace_cost']),
        transition_impulse=impulse, entry_pitch=entry_pitch,
        entry_simultaneous=order['simultaneous'].float(), entry_lead=order['lead'].float(),
        entry_support=support, entry_air=air,
        slip=slip, supported=supported, command=command[:, 0], straight=straight,
        flat=flat['active'], speed=speed)
    return env._v111_traction


def traction_quality(env):
    return state(env)['quality']


def traction_cost(env):
    return -state(env)['cost']


def pitch_cost(env):
    return -state(env)['pitch_cost']


def flat_quality(env):
    return state(env)['flat_quality']


def transition_impulse_penalty(env):
    return -state(env)['transition_impulse']


def entry_pitch_penalty(env):
    return -state(env)['entry_pitch']


def entry_simultaneous_penalty(env):
    return -state(env)['entry_simultaneous']/float(env.step_dt)


def entry_lead_reward(env):
    return state(env)['entry_lead']/float(env.step_dt)


def metric(env, env_ids, key):
    s = state(env)
    if key in ('partial_rear_slip', 'full_rear_slip'):
        speed_mask = (s['command'] > .1) & ((s['command'] < .85) if key.startswith('partial') else (s['command'] >= .85))
        mask = s['supported'][env_ids, 2:] & (s['straight'] & speed_mask)[env_ids, None]
        return (s['slip'][env_ids, 2:]*mask).sum()/mask.sum().clamp_min(1)
    if key == 'horizontal_support_fraction':
        return s['supported'][env_ids].float().mean()
    return s[key][env_ids].float().mean()
