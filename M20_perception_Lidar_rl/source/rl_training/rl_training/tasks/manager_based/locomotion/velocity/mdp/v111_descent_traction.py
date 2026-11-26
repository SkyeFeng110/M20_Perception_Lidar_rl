"""Opt-in V2: descending original-route tracking and coherent low-mu practice."""
import torch
from rl_training.route_hold import latch_reference
from rl_training.descent_traction import descent_route_terms, slip_tail_terms
from .v111_flat_posture import FlatPostureVelocityCommand
from . import v111_route_hold as parent
from .v135_terrain_geometry import column_names


def descent_name(name):
    return name == 'pyramid_stairs' or name.startswith('v2_descent_tread_')


def descent_mask(env):
    if not hasattr(env, '_m20_v2_descent_columns'):
        env._m20_v2_descent_columns = torch.tensor([descent_name(n) for n in column_names(
            env.cfg.scene.terrain.terrain_generator)], device=env.device)
    return env._m20_v2_descent_columns[env.scene.terrain.terrain_types]


class DescentRepairVelocityCommand(FlatPostureVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.turn_yaw = torch.zeros(self.num_envs, device=self.device)
        self.turn_forward = torch.zeros_like(self.turn_yaw)

    def _turn_cohort(self, ids):
        # 25% of flat environments; retain 62.5% straight profiles and
        # 12.5% native 3-D commands. Ascent/descent allocation is unchanged.
        profile = ids.remainder(16)
        return (self._flat_sequence_columns[self._env.scene.terrain.terrain_types[ids]]
                & (profile >= 4) & (profile <= 7))

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        from .v135_terrain_geometry import v135_terrain_geometry
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        up = ids[v135_terrain_geometry(self._env)['ascent'][ids]]
        up_speeds = self.vel_command_b.new_tensor([.3, .5, .7, 1.])
        self.refine_speed[up] = up_speeds[torch.randint(4, (len(up),), device=self.device)]
        direct = up[~self.refine_cohort[up]]
        self.vel_command_b[direct, 0] = self.refine_speed[direct]
        chosen = ids[descent_mask(self._env)[ids] & (ids.remainder(2) == 0)]
        speeds = self.vel_command_b.new_tensor([.3, .5, .7])
        self.refine_speed[chosen] = speeds[torch.randint(3, (len(chosen),), device=self.device)]
        self.vel_command_b[chosen] = 0
        self.is_heading_env[chosen] = False
        self.is_standing_env[chosen] = False
        turn = ids[self._turn_cohort(ids)]
        amplitude = self.vel_command_b.new_tensor([.2, .4, .6])
        self.turn_yaw[turn] = amplitude[torch.randint(3, (len(turn),), device=self.device)]*(
            2*torch.randint(2, (len(turn),), device=self.device)-1)
        self.turn_forward[turn] = .3*(torch.rand(len(turn), device=self.device) < .25)
        self.vel_command_b[turn] = 0
        self.is_heading_env[turn] = False
        self.is_standing_env[turn] = False

    def _update_command(self):
        super()._update_command()
        ids = torch.arange(self.num_envs, device=self.device)
        selected = descent_mask(self._env) & (ids.remainder(2) == 0)
        self.vel_command_b[selected, 0] = self.refine_speed[selected]*(self.refine_elapsed[selected] >= 1.)
        self.vel_command_b[selected, 1:] = 0
        from rl_training.front_turn import turn_profile
        turn = self._turn_cohort(ids)
        forward, yaw = turn_profile(self.refine_elapsed[turn], self.turn_yaw[turn], self.turn_forward[turn])
        self.vel_command_b[turn, 0], self.vel_command_b[turn, 2] = forward, yaw
        self.vel_command_b[turn, 1] = 0


def reset(env, env_ids):
    parent.reset(env, env_ids)
    if hasattr(env, '_m20_v2_descent'):
        env._m20_v2_descent['pending'][env_ids] = True
    if hasattr(env, '_m20_v2_slip'):
        env._m20_v2_slip['step'] = -1
    if hasattr(env, '_m20_front_turn'):
        env._m20_front_turn['pending'][env_ids] = True


def state(env):
    robot = env.scene['robot']
    if not hasattr(env, '_m20_v2_descent'):
        z = lambda *shape: torch.zeros((env.num_envs, *shape), device=env.device)
        env._m20_v2_descent = dict(step=-1, heading=z(), origin=z(2), valid=z().bool(),
            pending=torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
            length=env.episode_length_buf.clone(), samples=z(), sums=z(7), peaks=z(2))
    s = env._m20_v2_descent
    if s['step'] == int(env.common_step_counter):
        return s
    reset_mask = s['pending'] | (env.episode_length_buf < s['length']) | (env.episode_length_buf == 0)
    command = env.command_manager.get_command('base_velocity')
    term = env.command_manager.get_term('base_velocity')
    straight = (descent_mask(env) & (command[:, 0] >= 0)
                & (command[:, 1].abs() < .05) & (command[:, 2].abs() < .03) & ~term.is_heading_env)
    s['heading'], s['origin'], s['valid'] = latch_reference(s['heading'], s['origin'], s['valid'],
        robot.data.heading_w, robot.data.root_pos_w[:, :2], straight, reset_mask)
    heading = s['heading']
    yaw = robot.data.heading_w-heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    delta = robot.data.root_pos_w[:, :2]-s['origin']
    lateral = -delta[:, 0]*heading.sin()+delta[:, 1]*heading.cos()
    velocity = robot.data.root_lin_vel_w
    forward = velocity[:, 0]*heading.cos()+velocity[:, 1]*heading.sin()
    side = -velocity[:, 0]*heading.sin()+velocity[:, 1]*heading.cos()
    s.update(descent_route_terms(yaw, lateral, side, robot.data.root_ang_vel_w[:, 2],
                                forward, command[:, 0], straight))
    for key in ('samples', 'sums', 'peaks'):
        s[key][reset_mask] = 0
    active = s['active'] & ~reset_mask
    values = torch.stack((yaw.abs(), lateral.abs(), side.abs(), robot.data.root_ang_vel_w[:, 2].abs(),
                          forward, yaw, lateral), 1)
    s['samples'] += active
    s['sums'] += values*active[:, None]
    s['peaks'] = torch.maximum(s['peaks'], values[:, :2]*active[:, None])
    s.update(step=int(env.common_step_counter), length=env.episode_length_buf.clone())
    s['pending'][:] = False
    return s


def route_cost(env):
    return state(env)['cost']*float(getattr(env, '_m20_v2_descent_multiplier', 1.))


def route_quality(env):
    return state(env)['quality']*float(getattr(env, '_m20_v2_descent_multiplier', 1.))


def ascent_route_cost(env):
    s = parent.state(env)
    early = parent.early_traction_state(env)
    low = getattr(env, '_m20_v2_low_mu', torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    focus = early['early_window'].any(1) & early['eligible']
    bounded = lambda x: x.square()/(1+x.square())
    route_error = (.6*bounded((s['yaw']-.01745).clamp_min(0)/.05236)
                   +.4*bounded((s['lateral']-.025).clamp_min(0)/.075))
    # Stronger first-three-tread line keeping on coherent slippery stairs.
    # No extra forward-speed floor, leading-side bias or sagittal target.
    extra = .30*route_error*focus*(1.+.25*low)
    return s['cost']-extra


def outside_route_focus(env, reward_func, reward_params):
    if isinstance(reward_func, str):
        from isaaclab.utils import string_to_callable
        reward_func = string_to_callable(reward_func)
    return reward_func(env, **reward_params)*~(parent.state(env)['active'] | state(env)['active'])


def slip_state(env):
    from .v111_traction_transition import state as traction_state
    from rl_training.traction_transition import handoff_patch
    if hasattr(env, '_m20_v2_slip') and env._m20_v2_slip['step'] == int(env.common_step_counter):
        return env._m20_v2_slip
    old = parent.early_traction_state(env)
    traction = traction_state(env)
    # Loaded landing frames matter to P95. Horizontal center/adjacent plane,
    # actual >=15N force and bottom distance still exclude air/riser contacts.
    supported = handoff_patch(traction['wheel_bottom'], traction['patch_hits'],
        traction['patch_normals'], traction['force_z'], traction['vertical_speed'])
    reward, cost, selected = slip_tail_terms(traction['slip'], supported,
        old['early_window'], env._m20_58098_handoff['speed'], traction['command'])
    low = getattr(env, '_m20_v2_low_mu', torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    gain = (1.+.25*low)*float(getattr(env, '_m20_v2_slip_multiplier', 1.))
    s = dict(step=int(env.common_step_counter), reward=reward*gain, cost=cost*gain,
             selected=selected, slip=traction['slip'], low=low)
    env._m20_v2_slip = s
    return s


def slip_quality(env):
    return slip_state(env)['reward']


def slip_cost(env):
    return -slip_state(env)['cost']


def metric(env, env_ids, key):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if key.startswith('low_mu_'):
        s = slip_state(env)
        mask = s['selected'][ids] & s['low'][ids, None]
        if key == 'low_mu_samples':
            return mask.sum()
        if key == 'low_mu_slip':
            return (s['slip'][ids]*mask).sum()/mask.sum().clamp_min(1)
    s = state(env)
    if key == 'samples':
        return s['samples'][ids].sum()
    if key.endswith('_peak'):
        k = ('yaw', 'lateral').index(key[:-5])
        return s['peaks'][ids, k].max() if len(ids) else s['samples'].new_zeros(())
    k = ('yaw', 'lateral', 'side_speed', 'yaw_rate', 'speed', 'yaw_signed', 'lateral_signed').index(key)
    return s['sums'][ids, k].sum()/s['samples'][ids].sum().clamp_min(1)
