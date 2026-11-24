"""Prepare a reviewable source patch; no board access, builds or motor control."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO/'mujoco_validation/M20_NX真机部署项目_20261120/1515_deploy/releases/aos67100'


def prepare(output):
    relative = Path('src/M20_sdk_deploy/run_policy/m20_policy_runner.hpp')
    original = (SOURCE/relative).read_text()
    changed = original
    replacements = [
        ('#include "policy_alignment_trace.hpp"', '#include "policy_alignment_trace.hpp"\n#include "forward_command_ramp.hpp"'),
        ('    VecXf history_before_attempt_;', '    VecXf history_before_attempt_;\n    m20::ForwardCommandRamp forward_ramp_, forward_before_attempt_;'),
        ('    std::array<float,16> alignment_raw_action_{};',
         '    std::array<float,16> alignment_raw_action_{};\n    std::array<float,3> requested_velocity_{};'),
        ('        e.action_ready=alignment_ready_;e.wheel_stop_enabled=zero_command_wheel_stop_;',
         '        e.action_ready=alignment_ready_;e.wheel_stop_enabled=zero_command_wheel_stop_;\n        e.requested_velocity = requested_velocity_;'),
        ('        history_pending_ = false;\n    }\n    void RecordAlignmentInference',
         '        if (history_pending_ && !committed) forward_ramp_ = forward_before_attempt_;\n        history_pending_ = false;\n    }\n    void RecordAlignmentInference'),
        ('        cmd_vel_input_.setZero();', '        cmd_vel_input_.setZero();\n        forward_ramp_.reset();'),
        ('        history_before_attempt_ = last_action_eigen;',
         '        history_before_attempt_ = last_action_eigen;\n        forward_before_attempt_ = forward_ramp_;\n        requested_velocity_ = {uc.forward_vel_scale, uc.side_vel_scale, uc.turnning_vel_scale};'),
        ('        Vec3f command = Vec3f(uc.forward_vel_scale, uc.side_vel_scale, uc.turnning_vel_scale);',
         '        Vec3f command = Vec3f(forward_ramp_.sample(uc.forward_vel_scale, agent_timestep),\n                              uc.side_vel_scale, uc.turnning_vel_scale);')]
    for old, new in replacements:
        if changed.count(old) != 1:
            raise ValueError('Unexpected source context: '+old)
        changed = changed.replace(old, new)
    output.mkdir(parents=True, exist_ok=False)
    destination = output/'overlay'/relative
    destination.parent.mkdir(parents=True)
    destination.write_text(changed)
    header = REPO/'mujoco_validation/deploy_real/forward_command_ramp.hpp'
    shutil.copy2(header, destination.parent/header.name)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
        fromfile='a/'+str(relative), tofile='b/'+str(relative)))
    diff += ''.join(difflib.unified_diff([], header.read_text().splitlines(True),
        fromfile='/dev/null', tofile='b/'+str(relative.parent/header.name)))
    trace_relative = relative.parent/'policy_alignment_trace.hpp'
    original_trace = (SOURCE/trace_relative).read_text()
    changed_trace = original_trace
    for old, new in [
        ('    std::array<float,63> obs63{};', '    std::array<float,63> obs63{};\n    std::array<float,3> requested_velocity{};'),
        ('        j["event"]="policy_inference";j["received_6d"]=e.received_6d;',
         '        j["event"]="policy_inference";j["received_6d"]=e.received_6d;\n        j["requested_velocity3"]=e.requested_velocity;')]:
        if changed_trace.count(old) != 1: raise ValueError('Unexpected trace source context')
        changed_trace = changed_trace.replace(old, new)
    (output/'overlay'/trace_relative).write_text(changed_trace)
    diff += ''.join(difflib.unified_diff(original_trace.splitlines(True), changed_trace.splitlines(True),
        fromfile='a/'+str(trace_relative), tofile='b/'+str(trace_relative)))
    (output/'forward_command.patch').write_text(diff)
    receipt = dict(status='source_patch_prepared_not_deployed', source=str(SOURCE),
        source_runner_sha256=hashlib.sha256(original.encode()).hexdigest(),
        normal_forward_acceleration_m_s2=1.5, normal_forward_deceleration_m_s2=10./3.,
        max_forward_command_unchanged=True, reset_on_policy_enter=True,
        rollback_on_uncommitted_action=True, emergency_path_unchanged=True,
        requested_and_filtered_command_trace=True,
        actor_dimensions=[63, 16], board_connection=False, training_started=False,
        native_rebuild_required=True, offline_and_hardware_validation_pending=True)
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output), indent=2))
