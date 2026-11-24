// Offline only: no SDK/ROS publishers, no motor interface instantiated.
#include "m20_policy_runner.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        M20PolicyRunner runner("offline_contract_replay", argv[1]);
        runner.OnEnter();
        std::ifstream input(argv[2]);
        if (!input) return 2;
        std::string line;
        size_t count = 0;
        float maximum_error = 0;
        while (std::getline(input, line)) {
            std::istringstream row(line);
            RobotBasicState state;
            UserCommand command{};
            state.base_rot_mat.setIdentity();
            state.v11_observation.resize(6);
            VecXf expected_position(16), expected_velocity(16);
            for (int i=0; i<16; ++i) row >> state.joint_pos(i);
            for (int i=0; i<16; ++i) row >> state.joint_vel(i);
            for (int i=0; i<3; ++i) row >> state.base_omega(i);
            float forward=0, side=0, turn=0;
            row >> forward >> side >> turn;
            command.forward_vel_scale=forward;
            command.side_vel_scale=side;
            command.turnning_vel_scale=turn;
            for (int i=0; i<6; ++i) row >> state.v11_observation(i);
            for (int i=0; i<16; ++i) row >> expected_position(i);
            for (int i=0; i<16; ++i) row >> expected_velocity(i);
            if (!row) throw std::runtime_error("truncated fixture");
            state.v11_valid = true;
            state.v11_age_seconds = 0;
            const auto action = runner.getRobotAction(state, command);
            const float error = std::max((action.goal_joint_pos-expected_position).cwiseAbs().maxCoeff(),
                                         (action.goal_joint_vel-expected_velocity).cwiseAbs().maxCoeff());
            if (!action.goal_joint_pos.allFinite() || !action.goal_joint_vel.allFinite() || error > 0.002f)
                throw std::runtime_error("native controller parity failed row " + std::to_string(count) +
                                         " max_error=" + std::to_string(error));
            maximum_error = std::max(maximum_error, error);
            runner.FinalizeAction(true); // Commit offline history; no publication.
            ++count;
        }
        if (count != 128) return 1;
        std::cout << "NATIVE_CONTRACT_REPLAY=PASS samples=" << count
                  << " max_target_error=" << maximum_error << " scope=no_joint_commands\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
