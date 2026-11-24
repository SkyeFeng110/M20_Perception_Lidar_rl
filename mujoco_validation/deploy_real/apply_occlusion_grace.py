"""Apply bounded perception loss handling to the exact stopped, audited AOS source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from finalize_joint_guard_release import atomic_write, check_identity, validate
from generate_perception_grace import generate
from release_utils import replace_once, refresh_checksums

BASELINE = {
    'run_policy/v11_source_identity.hpp': 'd28ae2e9e292850211b3608d74c0bac9d01edeb4cd84cf4e249ce033854aec74',
    'run_policy/v11_observation_bundle.hpp': '546e9e1ed5a3ec214059533fd454ec7fc7bcc562cbd14f6fd1e8237065d06f37',
    'run_policy/v11_observation_store.hpp': '7526595e3e8e941803fee57bc899b43e14079b3b015f14b6d61153012aeb0d5d',
    'interface/robot/hardware/dds_interface.hpp': 'a8b4430f436b934670fb1302d8b18e6ef22778154c8539e8ce4698df1a55020e',
    'run_policy/m20_policy_runner.hpp': 'bfa8a4f419a5c856d6c0797e92dada7d1804b27dfbe757a5cfe2b0e890fb9acb',
    'state_machine/state_machine_base.h': 'a633c8879e4a2631c41ea627022fbe68fa5d8c88296d2cf9ec2a0d923ede15a7',
    'run_policy/policy_alignment_trace.hpp': 'ae78d2d03517a37cc44c51bce6677d95d8902a454c1a5268f61911ce96ece3b3',
    'CMakeLists.txt': '218dc00c06873fe30c27eb4d735a80beca54c6a823f236c155c93694bf14bd47',
}


def stage(source, payload):
    """Prepare every edit in memory before revoking release admission."""
    for name, expected in BASELINE.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Unknown AOS source: '+name)
    edits = {}
    for name in ('v11_source_identity.hpp', 'v11_observation_bundle.hpp', 'v11_observation_store.hpp'):
        edits['run_policy/'+name] = (payload/name).read_bytes()
    ledger = payload/'perception_grace_limits.json'
    with tempfile.TemporaryDirectory() as tmp:
        header = Path(tmp)/'perception_grace_limits.hpp'
        generate(ledger, header)
        edits['run_policy/perception_grace_limits.hpp'] = header.read_bytes()
    p = 'interface/robot/hardware/dds_interface.hpp'
    text = replace_once((source/p).read_text(), '#include "incident_recorder.hpp"',
                        '#include "incident_recorder.hpp"\n#include "perception_grace_limits.hpp"')
    text = replace_once(text, 'std::getenv("M20_V11_OBSERVATION_SOURCE"))};',
                        'std::getenv("M20_V11_OBSERVATION_SOURCE")), .3, m20::perception_limits::loss_timeout_s};')
    edits[p] = text.encode()
    p = 'run_policy/m20_policy_runner.hpp'
    text = replace_once((source/p).read_text(),
        '(!retained && !state.v11_observation.isZero(0)))',
        '(!retained && state.v11_lease.selection != m20::v11::FeatureSelection::DegradedHold && !state.v11_observation.isZero(0)))')
    text = replace_once(text, 'm20::incident::reason(r,alignment_ready_?"ready":"inference_not_ready");',
        'm20::incident::reason(r,alignment_ready_?(ro.v11_lease.selection==m20::v11::FeatureSelection::DegradedHold?"ready_perception_degraded":"ready"):"inference_not_ready");')
    edits[p] = text.encode()
    p = 'state_machine/state_machine_base.h'
    text = replace_once((source/p).read_text(), '    bool LatchV11FailureFor(StateName state) {',
        '    bool perception_degraded_notice_=false;\n    bool LatchV11FailureFor(StateName state) {')
    text = replace_once(text, '        if (observation.valid) return false;', '''        if (observation.valid) {
            const bool degraded=observation.lease.selection==m20::v11::FeatureSelection::DegradedHold;
            if (!degraded || IsHighGainState(current_state_name_)) {
                if (degraded!=perception_degraded_notice_) {
                    std::cerr << (degraded?"[PERCEPTION] Short loss: continuing policy with held terrain; age_s=":"[PERCEPTION] Fresh measured source restored; age_s=")
                              << observation.age_seconds << std::endl;
                    perception_degraded_notice_=degraded;
                }
                return false;
            }
            // Grace sustains existing motion; it never admits a new high-gain
            // transition from idle/damping without a fresh measured source.
        }''')
    edits[p] = text.encode()
    p = 'run_policy/policy_alignment_trace.hpp'
    text = replace_once((source/p).read_text(),
        '{"health_deadline_s",l.health_deadline_s},{"geometry_deadline_s",l.geometry_deadline_s},',
        '{"health_deadline_s",l.health_deadline_s},{"geometry_deadline_s",l.geometry_deadline_s},{"grace_deadline_s",l.grace_deadline_s},')
    edits[p] = text.encode()
    p = 'CMakeLists.txt'
    text = replace_once((source/p).read_text(), '  add_executable(joint_command_guard_test scripts/joint_command_guard_test.cpp)', '''  add_executable(perception_grace_test scripts/perception_grace_test.cpp)
  add_executable(perception_grace_worker_test scripts/perception_grace_worker_test.cpp ${STATE_MACHINE_SRC})
  target_compile_definitions(perception_grace_worker_test PRIVATE M20_FSM_NO_MOTOR_TEST=1)
  ament_target_dependencies(perception_grace_worker_test rclcpp drdds std_msgs sensor_msgs)
  target_link_libraries(perception_grace_worker_test onnxruntime -lpthread -ldl)
  install(TARGETS perception_grace_test perception_grace_worker_test DESTINATION lib/${PROJECT_NAME})
  add_executable(joint_command_guard_test scripts/joint_command_guard_test.cpp)''')
    edits[p] = text.encode()
    for name in ('perception_grace_test.cpp', 'perception_grace_worker_test.cpp'):
        edits['scripts/'+name] = (payload/name).read_bytes()
    return edits, ledger.read_bytes()


def apply(root, payload, mac, serial):
    check_identity(mac, serial)
    if os.geteuid() != 0:
        raise ValueError('Root required to inspect all controllers')
    manifest, _ = validate(root)
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:
            executable = p.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if executable.name == 'rl_deploy' and executable != Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
            raise ValueError('Custom motor controller active')
    source = root/'src/M20_sdk_deploy'
    edits, ledger = stage(source, payload)
    manifest.update(native_rebuild_required=True, offline_validation_complete=False,
                    joint_guard_finalized=False, perception_grace_revision='bounded_loss_v1',
                    perception_grace_ledger_sha256=hashlib.sha256(ledger).hexdigest())
    atomic_write(root/'AOS_RELEASE_MANIFEST.json', (json.dumps(manifest, indent=2)+'\n').encode(), 0o644)
    atomic_write(root/'perception_grace_limits.json', ledger, 0o644)
    for name, content in edits.items():
        atomic_write(source/name, content, 0o644)
    refresh_checksums(root)
    print('PERCEPTION_GRACE_WIRED scope=not_built_not_started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--expected-mac', required=True)
    parser.add_argument('--expected-serial', required=True)
    args = parser.parse_args()
    apply(args.release.resolve(), args.sources.resolve(), args.expected_mac, args.expected_serial)
