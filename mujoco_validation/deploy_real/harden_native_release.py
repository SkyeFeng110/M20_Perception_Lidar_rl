"""Wire safety into an isolated copy of the audited, final 42900 AOS release.

Does not start a controller. Refuses unknown source hashes and repeated patching.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from generate_joint_safety import generate
from release_utils import replace_once, refresh_checksums

HERE = Path(__file__).resolve().parent
BASELINE_MANIFEST_SHA256 = 'a06c5b4052dec32215cdc7e2f23ddaa58f944d7c6bdba3d7f18fb3bb4c79977d'
SOURCE_NAMES = ['interface/robot/hardware/dds_interface.hpp', 'interface/robot/robot_interface.h',
                'run_policy/m20_policy_runner.hpp', 'state_machine/state_machine_base.h',
                'state_machine/quadruped_wheel/idle_state.hpp', 'state_machine/quadruped_wheel/joint_damping_state.hpp',
                'state_machine/parameters/m20_control_parameters.cpp', 'CMakeLists.txt']


def refresh_from(release, baseline, baseline_manifest):
    # Rebuild a generated patch only after checking both the currently staged
    # release and the original. Unknown local edits are never overwritten.
    for root, sums in [(release, release/'RELEASE_SHA256SUMS'), (baseline, baseline_manifest)]:
        for line in sums.read_text().splitlines():
            expected, name = line.split(None, 1)
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Cannot refresh changed release file: '+str(root/name))
    for name in SOURCE_NAMES:
        shutil.copy2(baseline/'src/M20_sdk_deploy'/name, release/'src/M20_sdk_deploy'/name)
    shutil.copy2(baseline/'AOS_RELEASE_MANIFEST.json',release/'AOS_RELEASE_MANIFEST.json')


def harden(release, baseline_manifest):
    if hashlib.sha256(baseline_manifest.read_bytes()).hexdigest() != BASELINE_MANIFEST_SHA256:
        raise ValueError('Not the audited final 42900 manifest')
    expected = dict((p, h) for h, p in (line.split(None, 1) for line in baseline_manifest.read_text().splitlines()))
    src = release/'src/M20_sdk_deploy'
    for name in SOURCE_NAMES:
        path = src/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['src/M20_sdk_deploy/'+name]:
            raise ValueError('Source differs from audited final release: '+name)
    # Also require the actual checkpoint export, not just matching SDK source.
    if hashlib.sha256((release/'policy/model.onnx').read_bytes()).hexdigest() != expected['policy/model.onnx']:
        raise ValueError('Policy differs from baseline')
    for name in ('joint_command_guard.hpp', 'incident_recorder.hpp', 'guarded_dds_publish.inc'):
        shutil.copy2(HERE/name, src/'run_policy'/name)
    shutil.copy2(HERE/'joint_safety_limits.json', release/'joint_safety_limits.json')
    generate(release/'joint_safety_limits.json', src/'run_policy/joint_safety_limits.hpp')
    p=src/'interface/robot/robot_interface.h'
    text=replace_once(p.read_text(),'    virtual MatXf GetJointCommand(){',
        '    virtual bool HasJointCommandFault() const {return false;}\n'
        '    virtual void RecordControllerState(int,const UserCommand&) {}\n\n    virtual MatXf GetJointCommand(){')
    p.write_text(text)
    p=src/'interface/robot/hardware/dds_interface.hpp'
    text=replace_once(p.read_text(),'#include "policy_alignment_trace.hpp"',
        '#include "policy_alignment_trace.hpp"\n#include "joint_command_guard.hpp"\n#include "incident_recorder.hpp"')
    text=replace_once(text,'    double time_stamp_;','''    mutable std::mutex joint_publish_mutex_;
    m20::joint_safety::Guard joint_guard_;
    std::atomic<bool> joint_command_fault_{false};
    double joint_receipt_mono_s_=0;
    void EmitJointMessage(const drdds::msg::JointsDataCmd& msg) {
#ifdef M20_DDS_NO_MOTOR_TEST
        if(!test_joint_sink_)throw std::logic_error("no-motor test sink missing");
        test_joint_sink_(msg);
#else
        joint_cmd_pub_->publish(msg);
#endif
    }
#ifdef M20_DDS_NO_MOTOR_TEST
    std::function<void(const drdds::msg::JointsDataCmd&)> test_joint_sink_;
#endif
    double time_stamp_;''')
    text=replace_once(text,'        (void)m20::alignment::Journal::global(); // allocate/open before control starts',
        '''        if(dof_num!=16)throw std::runtime_error("Joint safety requires M20 16-DOF ABI");
        if(!m20::joint_limits::operator_limits_confirmed)throw std::runtime_error("Operator joint bounds not confirmed");
        if(!m20::incident::Recorder::global().healthy())throw std::runtime_error("Always-on incident recorder required for hardware interface");
        (void)m20::alignment::Journal::global(); // allocate/open before control starts''')
    # Control-word reset handshake has no gains/targets. It also uses the
    # no-motor sink in the test executable; no test can call a DDS publisher.
    text=replace_once(text,'            joint_cmd_pub_->publish(msg);','            EmitJointMessage(msg);')
    text=replace_once(text,'        joint_cmd_pub_ = node_->create_publisher<drdds::msg::JointsDataCmd>("/JOINTS_CMD", 10);',
        '#ifndef M20_DDS_NO_MOTOR_TEST\n        joint_cmd_pub_ = node_->create_publisher<drdds::msg::JointsDataCmd>("/JOINTS_CMD", 10);\n#endif')
    begin=text.index('    void PublishJointMessage(')
    end=text.index('\n    virtual VecXf GetMotorTemperture()',begin)
    text=text[:begin]+'#include "guarded_dds_publish.inc"\n'+text[end:]
    anchor='''        ri_ts_.store(rclcpp::Time(msg->header.stamp).seconds(), std::memory_order_release);
    }'''
    text=replace_once(text,anchor,'''        ri_ts_.store(rclcpp::Time(msg->header.stamp).seconds(), std::memory_order_release);
        joint_receipt_mono_s_=m20::incident::mono();
        m20::incident::Record record;record.kind=m20::incident::Kind::Feedback;
        record.mono_s=joint_receipt_mono_s_;record.wall_s=m20::incident::wall();record.feedback_mono_s=joint_receipt_mono_s_;
        for(int i=0;i<16;++i){record.q[i]=joint_pos_(i);record.qd[i]=joint_vel_(i);record.tau[i]=joint_tau_(i);}
        m20::incident::Recorder::global().put(record);
    }''')
    p.write_text(text)
    p=src/'state_machine/state_machine_base.h';text=p.read_text()
    text=replace_once(text,'                ri_ptr_->RefreshRobotData();','''                ri_ptr_->RefreshRobotData();
                ri_ptr_->RecordControllerState(int(current_state_name_),*uc_ptr_->GetUserCommand());''')
    text=replace_once(text,'                const bool force_damping = command->safe_control_mode != 0 ||',
        '                const bool force_damping = ri_ptr_->HasJointCommandFault() || command->safe_control_mode != 0 ||')
    text=replace_once(text,'                if (LatchV11FailureFor(next_state_name_)) {',
        '                if (ri_ptr_->HasJointCommandFault() || LatchV11FailureFor(next_state_name_)) {')
    p.write_text(text)
    for name in ('idle_state.hpp','joint_damping_state.hpp'):
        p=src/'state_machine/quadruped_wheel'/name
        text=replace_once(p.read_text(),'if (data_ptr_->perception_fault_latched.load(std::memory_order_acquire)) {',
            'if (ri_ptr_->HasJointCommandFault() || data_ptr_->perception_fault_latched.load(std::memory_order_acquire)) {')
        if name=='idle_state.hpp':
            text=replace_once(text,'#include "state_base.h"','#include "state_base.h"\n#include "joint_safety_limits.hpp"')
            begin=text.index('            Vec4f fl_lower, fl_upper;');end=text.index('            for (int i = 0; i < 16; ++i) {',begin)
            text=text[:begin]+'''            for(int i=0;i<16;++i){
                joint_pos_lower(i)=i%4==3?-INFINITY:m20::joint_limits::lower(i);
                joint_pos_upper(i)=i%4==3?INFINITY:m20::joint_limits::upper(i);
            }
'''+text[end:]
        p.write_text(text)
    p=src/'run_policy/m20_policy_runner.hpp';text=p.read_text()
    text=replace_once(text,'#include "trained_model_contract.hpp"','#include "trained_model_contract.hpp"\n#include "incident_recorder.hpp"')
    text=replace_once(text,'    void RecordAlignmentInference(const RobotBasicState&ro)const{','''    void RecordAlignmentInference(const RobotBasicState&ro)const{
        auto& recorder=m20::incident::Recorder::global();
        if(recorder.enabled()){
            m20::incident::Record r;r.kind=m20::incident::Kind::Inference;
            r.mono_s=m20::incident::mono();r.wall_s=m20::incident::wall();r.inference_id=alignment_inference_id_;
            r.observation_valid=ro.v11_valid;r.source_age_s=ro.v11_age_seconds;r.deadline_s=ro.v11_steady_deadline_seconds;
            r.source_seq=ro.v11_source.source_seq;r.publish_seq=ro.v11_source.publish_seq;
            std::strncpy(r.epoch,ro.v11_source.epoch.data(),sizeof(r.epoch)-1);
            if(alignment_observation_built_)std::copy_n(current_observation_.data(),63,r.obs.begin());
            r.inferred=alignment_inferred_;
            if(r.inferred){r.raw_action=alignment_raw_action_;std::copy_n(current_action_eigen.data(),16,r.processed_action.begin());}
            m20::incident::reason(r,alignment_ready_?"ready":"inference_not_ready");recorder.put(r);
        }''')
    p.write_text(text)
    # Existing controls consume their constants from the same ledger. Effective
    # per-leg asymmetric bounds are enforced independently by the final guard.
    p=src/'state_machine/parameters/m20_control_parameters.cpp';text=p.read_text()
    text=replace_once(text,'#include "control_parameters.h"','#include "control_parameters.h"\n#include "joint_safety_limits.hpp"')
    begin=text.index('    fl_joint_lower_ <<');end=text.index('\n}',begin)
    text=text[:begin]+'''    for(int i=0;i<3;++i){
        fl_joint_lower_(i)=m20::joint_limits::lower(i);
        fl_joint_upper_(i)=m20::joint_limits::upper(i);
        joint_vel_limit_(i)=m20::joint_limits::leg_velocity_rad_s[i];
        torque_limit_(i)=m20::joint_limits::leg_torque_nm[i];
    }'''+text[end:];p.write_text(text)
    for name in ('joint_command_guard_test.cpp','incident_recorder_test.cpp','dds_joint_guard_no_motor_test.cpp'):
        shutil.copy2(HERE/name,src/'scripts'/name)
    p=src/'CMakeLists.txt';text=p.read_text()
    text=replace_once(text,'  add_executable(rl_deploy main.cpp ${STATE_MACHINE_SRC})', '''  add_executable(joint_command_guard_test scripts/joint_command_guard_test.cpp)
  add_executable(incident_recorder_test scripts/incident_recorder_test.cpp)
  target_link_libraries(incident_recorder_test -lpthread)
  add_executable(dds_joint_guard_no_motor_test scripts/dds_joint_guard_no_motor_test.cpp)
  target_compile_definitions(dds_joint_guard_no_motor_test PRIVATE M20_DDS_NO_MOTOR_TEST=1 M20_GOS_ONLY_CONTROL=1)
  ament_target_dependencies(dds_joint_guard_no_motor_test rclcpp drdds std_msgs sensor_msgs)
  target_link_libraries(dds_joint_guard_no_motor_test -lpthread)
  install(TARGETS joint_command_guard_test incident_recorder_test dds_joint_guard_no_motor_test DESTINATION lib/${PROJECT_NAME})
  add_executable(rl_deploy main.cpp ${STATE_MACHINE_SRC})''')
    p.write_text(text)
    shutil.copy2(HERE/'launch_aos.sh',release/'launch_aos.sh')
    manifest=json.loads((release/'AOS_RELEASE_MANIFEST.json').read_text())
    manifest.update(native_rebuild_required=True,offline_validation_complete=False,ready_for_motor_control=False,
        joint_safety_revision='joint_publish_guard_v1',joint_safety_ledger_sha256=hashlib.sha256((release/'joint_safety_limits.json').read_bytes()).hexdigest(),
        hardware_locomotion_validation='not_performed',control_started_at_packaging=False)
    (release/'AOS_RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    refresh_checksums(release)
    print('JOINT_GUARD_WIRED scope=not_built_not_started')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path)
    p.add_argument('--baseline-manifest',type=Path,required=True)
    p.add_argument('--refresh-from',type=Path)
    a=p.parse_args()
    if hashlib.sha256(a.baseline_manifest.read_bytes()).hexdigest() != BASELINE_MANIFEST_SHA256:
        raise ValueError('Not the audited final 42900 manifest')
    if a.refresh_from:refresh_from(a.release.resolve(),a.refresh_from.resolve(),a.baseline_manifest.resolve())
    harden(a.release.resolve(),a.baseline_manifest.resolve())
