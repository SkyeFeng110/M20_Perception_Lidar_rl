"""Generate deterministic SavedPolicy -> native controller comparison cases."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'mujoco_validation/deploy/mujoco_v135'))
    from policy import SavedPolicy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = SavedPolicy(args.checkpoint, args.config_dir)
    rng = np.random.default_rng(42900)
    permutation = [i for leg in range(4) for i in (3*leg, 3*leg+1, 3*leg+2, 12+leg)]
    previous = np.zeros(16)
    rows = []
    for index in range(128):
        position = policy.nominal + rng.uniform(-.25, .25, 16)
        velocity = rng.uniform(-15, 15, 16)
        omega = rng.uniform(-2, 2, 3)
        command = np.array([.2, 0., 0.]) if index % 4 else np.zeros(3)
        if index % 8 == 1:
            command = np.array([0., .15, .1])
        if index % 16 == 2:
            velocity[::2] = 150.  # Exercise saved joint-velocity clipping.
            omega[0] = 120.       # Exercise clipping before angular scaling.
        feature = np.array([1., -.16, .3, 1., .16, .55])
        selected_feature = feature if command[0] > .05 else np.zeros(6)
        obs = policy.observation(omega, np.array([0., 0., -1.]), command,
                                 position, velocity, previous, selected_feature)
        action = policy(obs)
        if np.linalg.norm(command) < .08:
            action[12:] = 0.
        previous = action.copy()
        targets = policy.nominal + action * policy.action_scale
        expected_position = np.zeros(16)
        expected_velocity = np.zeros(16)
        for robot, pol in enumerate(permutation):
            if robot % 4 == 3:
                expected_velocity[robot] = targets[pol]
            else:
                expected_position[robot] = targets[pol]
        rows.append(np.concatenate((position[permutation], velocity[permutation], omega,
                                    command, feature, expected_position, expected_velocity)))
    np.savetxt(args.output, rows, fmt='%.9g')
    print('NATIVE_FIXTURE=128 includes nominal/order/scaling/clipping/history/wheel_stop')


if __name__ == '__main__':
    main()
