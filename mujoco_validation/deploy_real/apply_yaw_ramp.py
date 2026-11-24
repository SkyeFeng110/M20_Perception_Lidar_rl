"""Apply turn onset ramp to the stopped, validated 1515 control worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from finalize_joint_guard_release import check_identity,validate,atomic_write
from generate_yaw_limits import generate
from release_utils import refresh_checksums,replace_once

WORKER='state_machine/quadruped_wheel/rl_control_state.hpp'
PINS={WORKER:'f38d11e1763a381be7eda48eebe285f38fa85774ca9088c93d8b968703159c63',
      'CMakeLists.txt':'6a4d41bd51592b4f51d1b7e16488aba92bb7631de6098968325d9fe867579449'}


def stage(source,payload):
    for name,expected in PINS.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Unknown source: '+name)
    text=replace_once((source/WORKER).read_text(),'#include "m20_policy_runner.hpp"',
                     '#include "m20_policy_runner.hpp"\n#include "yaw_command_ramp.hpp"')
    text=replace_once(text,'        float policy_cost_time_ = 1;', '''        float policy_cost_time_ = 1;
        m20::YawCommandRamp yaw_ramp_;
        UserCommand policy_user_command_;
#ifdef M20_FSM_NO_MOTOR_TEST
        std::atomic<float> applied_yaw_diagnostic_{0};
#endif''')
    text=replace_once(text,'                    auto ra = policy_ptr_->getRobotAction(observation, *(uc_ptr_->GetUserCommand()));', '''                    const auto* requested=uc_ptr_->GetUserCommand();
                    policy_user_command_.forward_vel_scale=requested->forward_vel_scale.load();
                    policy_user_command_.side_vel_scale=requested->side_vel_scale.load();
                    policy_user_command_.turnning_vel_scale=yaw_ramp_.sample(
                        requested->turnning_vel_scale.load(),m20::alignment::mono());
                    auto ra = policy_ptr_->getRobotAction(observation, policy_user_command_);
#ifdef M20_FSM_NO_MOTOR_TEST
                    const auto actual_input=m20_policy_->LastObservationForDiagnostics();
                    if(actual_input.size()==63)applied_yaw_diagnostic_.store(actual_input(8));
#endif''')
    text=replace_once(text,'            policy_ptr_->OnEnter();',
                     '            policy_ptr_->OnEnter();\n            yaw_ramp_.reset(m20::alignment::mono());')
    text=replace_once(text,'        VecXf LastObservationAfterExitForDiagnostics() const {',
                     '        float AppliedYawForDiagnostics()const{return applied_yaw_diagnostic_.load();}\n        VecXf LastObservationAfterExitForDiagnostics() const {')
    edits={WORKER:text.encode()}
    text=replace_once((source/'CMakeLists.txt').read_text(),
        '  add_executable(perception_grace_test scripts/perception_grace_test.cpp)',
        '  add_executable(yaw_command_ramp_test scripts/yaw_command_ramp_test.cpp)\n'
        '  install(TARGETS yaw_command_ramp_test DESTINATION lib/${PROJECT_NAME})\n'
        '  add_executable(perception_grace_test scripts/perception_grace_test.cpp)')
    edits['CMakeLists.txt']=text.encode()
    for name,dest in [('yaw_command_ramp.hpp','run_policy'),('yaw_command_ramp_test.cpp','scripts'),
                      ('perception_grace_worker_test.cpp','scripts')]:
        edits[dest+'/'+name]=(payload/name).read_bytes()
    ledger=(payload/'yaw_command_limits.json').read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        header=Path(temp)/'yaw_command_limits.hpp';generate(payload/'yaw_command_limits.json',header)
        edits['run_policy/yaw_command_limits.hpp']=header.read_bytes()
    return edits,ledger


def apply(root,payload,mac,serial):
    check_identity(mac,serial)
    if os.geteuid()!=0:raise ValueError('Root required to inspect all controllers')
    manifest,_=validate(root)
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:exe=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe.name=='rl_deploy' and exe!=Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
            raise ValueError('Custom controller active')
    edits,ledger=stage(root/'src/M20_sdk_deploy',payload)
    manifest.update(native_rebuild_required=True,offline_validation_complete=False,joint_guard_finalized=False,
                    yaw_ramp_revision='yaw_onset_v1',yaw_ledger_sha256=hashlib.sha256(ledger).hexdigest())
    atomic_write(root/'AOS_RELEASE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644)
    atomic_write(root/'yaw_command_limits.json',ledger,0o644)
    for name,data in edits.items():atomic_write(root/'src/M20_sdk_deploy'/name,data,0o644)
    refresh_checksums(root)
    print('YAW_RAMP_WIRED scope=not_built_not_started')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path)
    p.add_argument('--sources',required=True,type=Path);p.add_argument('--expected-mac',required=True);p.add_argument('--expected-serial',required=True)
    a=p.parse_args();apply(a.release.resolve(),a.sources.resolve(),a.expected_mac,a.expected_serial)
