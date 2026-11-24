"""Positive route quality, using privileged reward state only."""
import torch
from rl_training.step_retention import forward_route_quality


def route_reward(env):
    from .v111_step_support import state
    from .v135_terrain_geometry import v135_terrain_geometry
    state(env)
    robot = env.scene['robot']
    term = env.command_manager.get_term('base_velocity')
    command = env.command_manager.get_command('base_velocity')
    heading = term.v11_path_heading_w
    yaw = robot.data.heading_w-heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    offset = robot.data.root_pos_w[:, :2]-term.v11_path_origin_w
    lateral = -offset[:, 0]*heading.sin()+offset[:, 1]*heading.cos()
    speed = robot.data.root_lin_vel_w[:, 0]*heading.cos()+robot.data.root_lin_vel_w[:, 1]*heading.sin()
    lateral_speed = -robot.data.root_lin_vel_w[:, 0]*heading.sin()+robot.data.root_lin_vel_w[:, 1]*heading.cos()
    straight = ((command[:, 1].abs() < .05) & (command[:, 2].abs() < .03)
                & ~term.is_heading_env)
    active = v135_terrain_geometry(env)['ascent'] & ~env._v111_first_tread['entry_near'] & straight
    return forward_route_quality(speed, command[:, 0], yaw, lateral,
                                 robot.data.root_ang_vel_b[:, 2], lateral_speed)*active
