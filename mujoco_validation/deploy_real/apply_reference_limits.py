"""Stage the user-selected earlier software limits on a stopped guarded release.

Never starts a controller. Archives the prior ELF, inputs and checksums first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
from finalize_joint_guard_release import check_identity,validate,atomic_write
from generate_joint_safety import generate as generate_joint
from generate_policy_target_limits import generate as generate_target
from release_utils import refresh_checksums,replace_once

WORKER='state_machine/quadruped_wheel/rl_control_state.hpp'
FIXTURE='scripts/staircase_fsm_no_motor_test.cpp'
PINS={WORKER:'3467c855a5b17e3b937cc4960e004ae3f3e718188fad183c8d6c277984891bbc',
      FIXTURE:'164d99aab03222c2a0bc07163b95f2e1660c5e230af417d24778f701fc75d673',
      'CMakeLists.txt':'98c1ce7d62b202ae044bed6f0d0c30929ddffbbac8ce1bb6e35f2375117dfa5c'}


def stage(source,payload):
    for name,expected in PINS.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Unknown source: '+name)
    text=replace_once((source/WORKER).read_text(),'#include "yaw_command_ramp.hpp"',
                     '#include "yaw_command_ramp.hpp"\n#include "policy_target_limiter.hpp"')
    text=replace_once(text,'        m20::YawCommandRamp yaw_ramp_;',
                     '        m20::YawCommandRamp yaw_ramp_;\n        m20::joint_safety::PolicyTargetLimiter target_limiter_;')
    text=replace_once(text,'                            } else {\n                                if (v11_required) {',
        '''                            } else {
                                const auto shaped=target_limiter_.project(
                                    m20::joint_safety::PolicyTargetLimiter::from_matrix(res),now);
                                m20::joint_safety::PolicyTargetLimiter::to_matrix(shaped,res);
                                if (v11_required) {''')
    text=replace_once(text,'                                    decision_reason="policy_command_committed";\n                                }\n                            }',
        '''                                    decision_reason="policy_command_committed";
                                }
                                if(alignment_committed)target_limiter_.committed(
                                    m20::joint_safety::PolicyTargetLimiter::from_matrix(ri_ptr_->GetJointCommand()),
                                    m20::alignment::mono());
                            }''')
    text=replace_once(text,'            yaw_ramp_.reset(m20::alignment::mono());',
        '''            yaw_ramp_.reset(m20::alignment::mono());
            target_limiter_.seed(
                m20::joint_safety::PolicyTargetLimiter::from_matrix(ri_ptr_->GetJointCommand()),
                m20::alignment::mono());''')
    edits={'src/M20_sdk_deploy/'+WORKER:text.encode()}
    text=replace_once((source/FIXTURE).read_text(),
        '    std::function<void()> BeforePublisherGuardForDiagnostics;',
        '''    MatXf GetJointCommand() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!submissions.empty())return submissions.back().command;
        MatXf seed=MatXf::Zero(16,5);seed.col(1)=q_;
        for(int i=0;i<16;++i){seed(i,0)=i%4==3?0:80;seed(i,2)=i%4==3?.6f:2.f;}
        return seed; // Synthetic issued support for direct-worker fixtures only.
    }
    std::function<void()> BeforePublisherGuardForDiagnostics;''')
    edits['src/M20_sdk_deploy/'+FIXTURE]=text.encode()
    text=replace_once((source/'CMakeLists.txt').read_text(),
        '  add_executable(yaw_command_ramp_test scripts/yaw_command_ramp_test.cpp)',
        '  add_executable(policy_target_limiter_test scripts/policy_target_limiter_test.cpp)\n'
        '  install(TARGETS policy_target_limiter_test DESTINATION lib/${PROJECT_NAME})\n'
        '  add_executable(yaw_command_ramp_test scripts/yaw_command_ramp_test.cpp)')
    edits['src/M20_sdk_deploy/CMakeLists.txt']=text.encode()
    for name,dest in [('policy_target_limiter.hpp','run_policy'),('policy_target_limiter_test.cpp','scripts'),
                      ('perception_grace_worker_test.cpp','scripts')]:
        edits['src/M20_sdk_deploy/'+dest+'/'+name]=(payload/name).read_bytes()
    for name in ('joint_safety_limits.json','policy_target_limits.json'):
        edits[name]=(payload/name).read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        header=Path(temp)/'joint_safety_limits.hpp';generate_joint(payload/'joint_safety_limits.json',header)
        edits['src/M20_sdk_deploy/run_policy/joint_safety_limits.hpp']=header.read_bytes()
        header=Path(temp)/'policy_target_limits.hpp';generate_target(payload/'policy_target_limits.json',header)
        edits['src/M20_sdk_deploy/run_policy/policy_target_limits.hpp']=header.read_bytes()
    for name in ('build_1516_aos.sh','finalize_joint_guard_release.py','release_utils.py'):
        edits['validation_tools/'+name]=(payload/name).read_bytes()
    return edits


def apply(root,payload,mac,serial):
    check_identity(mac,serial)
    if os.geteuid()!=0:raise ValueError('Root required to inspect custom controllers')
    manifest,_=validate(root)
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:exe=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe.name=='rl_deploy' and exe!=Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
            raise ValueError('Custom controller active')
    edits=stage(root/'src/M20_sdk_deploy',payload)
    backup=root/'artifacts'/('reference_limits_before_'+time.strftime('%Y%m%d_%H%M%S')+'.tar.gz')
    names=set(edits)|{'AOS_RELEASE_MANIFEST.json','RELEASE_SHA256SUMS',
                     'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy'}
    with tarfile.open(backup,'x:gz') as tar:
        for name in sorted(names):
            if (root/name).is_file():tar.add(root/name,arcname=name)
    manifest.update(native_rebuild_required=True,offline_validation_complete=False,
                    joint_guard_finalized=False,position_limits_revision='prior_workspace_named_v1',
                    policy_target_revision='prior_workspace_projection_v1',
                    joint_safety_ledger_sha256=hashlib.sha256(edits['joint_safety_limits.json']).hexdigest(),
                    policy_target_ledger_sha256=hashlib.sha256(edits['policy_target_limits.json']).hexdigest(),
                    ready_for_motor_control=False,hardware_locomotion_validation='not_performed',
                    weights_only_update=False,
                    reference_limits_backup=str(backup))
    # Invalidate the old admission before changing any compiled input.
    atomic_write(root/'AOS_RELEASE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644)
    for name,data in edits.items():atomic_write(root/name,data,0o644)
    refresh_checksums(root)
    print('REFERENCE_LIMITS_STAGED scope=not_built_not_started backup='+str(backup))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path)
    p.add_argument('--sources',required=True,type=Path);p.add_argument('--expected-mac',required=True);p.add_argument('--expected-serial',required=True)
    a=p.parse_args();apply(a.release.resolve(),a.sources.resolve(),a.expected_mac,a.expected_serial)
