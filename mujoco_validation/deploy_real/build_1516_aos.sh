#!/usr/bin/env bash
# Offline compilation/tests only. Never executes aos-run or creates motor I/O.
set -euo pipefail
release="$(cd "${1:?Usage: build_1516_aos.sh RELEASE}" && pwd)"
tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$release"
[[ "$(uname -m)" == aarch64 ]]
sha256sum -c --quiet RELEASE_SHA256SUMS
# Source: actual 1516 common.hpp budget(): 4 GiB/compiler + 1 GiB reserve.
available_kib="$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)"
(( available_kib > 9*1024*1024 ))
(( $(df -Pk "$release" | awk 'NR==2 {print $4}') > 512*1024 ))
free -m
cat /proc/loadavg
set +u
source /opt/ros/foxy/setup.bash
set -u
export CMAKE_PREFIX_PATH="$release/install/drdds:${CMAKE_PREFIX_PATH:-}"
export AMENT_PREFIX_PATH="$release/install/drdds:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$release/install/drdds/lib:$release/src/M20_sdk_deploy/third_party/onnxruntime/arm/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$release/src/M20_sdk_deploy" -B "$release/build/m20_sdk_deploy" \
  -DCMAKE_INSTALL_PREFIX="$release/install/m20_sdk_deploy" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PLATFORM=arm -DBUILD_REALSENSE_DEPTH_SOURCE=OFF \
  -DM20_GOS_ONLY_CONTROL=ON -DBUILD_LEGACY_V11_BRIDGES=OFF
# Spare AOS control cores, normal scheduler, low priority; vendor keeps its cores.
nice -n 19 taskset -c 6,7 prlimit --as=4294967296 -- \
  cmake --build "$release/build/m20_sdk_deploy" --parallel 2
cmake --install "$release/build/m20_sdk_deploy"
python3 "$tools/release_utils.py" "$release"
export LD_LIBRARY_PATH="$release/install/m20_sdk_deploy/lib:$LD_LIBRARY_PATH"
export M20_V11_FEATURE_CONTRACT=v11_signed_edges_v2
bin="$release/install/m20_sdk_deploy/lib/m20_sdk_deploy"
validation_dir="$release/artifacts/validation_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$validation_dir"
# Explicit recovery uses the actual FSM and policy worker, with motor/UDP
# factories compiled out. DDS release and recorder restart are tested below.
python3 - "$validation_dir/recovery_fixture.json" <<'PY'
import json,sys
with open(sys.argv[1],'w') as f:
    json.dump({'robot':{'joint_pos':[0.]*16,'joint_vel':[0.]*16,'rpy':[0.]*3,'omega':[0.]*3},'recorded_6d':[0.]*6},f)
PY
export M20_POLICY_PATH="$release/policy/model.onnx"
for name in rl_explicit_recovery joint_explicit_recovery; do
  "$release/build/m20_sdk_deploy/staircase_fsm_no_motor_test" "$name" "$validation_dir/recovery_fixture.json" "$validation_dir/$name.json"
done
for name in staircase_packet_test v11_observation_store_test v11_riser_extractor_test \
            gamepad_parse_test stand_motion_guard_test policy_alignment_trace_test staircase_preflight_health_test; do
  "$bin/$name"
done
if [[ -f "$release/yaw_command_limits.json" ]]; then
  "$bin/yaw_command_ramp_test"
  "$bin/perception_grace_worker_test" yaw_ramp "$release/policy/model.onnx"
fi
if [[ -f "$release/policy_target_limits.json" ]]; then
  "$bin/policy_target_limiter_test"
  "$bin/perception_grace_worker_test" target_limits "$release/policy/model.onnx"
fi
if [[ -f "$release/perception_grace_limits.json" ]]; then
  "$bin/perception_grace_test"
  for name in ascent descent flat recovery hard_fault; do
    "$bin/perception_grace_worker_test" "$name" "$release/policy/model.onnx"
  done
fi
"$bin/policy_smoke_test" "$release/policy/model.onnx"
"$bin/policy_bundle_fixture" "$release/policy/model.onnx"
for name in expire_commit expire_selection expire_run fault_recovery publisher_expire publisher_fault; do
  "$bin/policy_bundle_worker_test" "$name" "$release/policy/model.onnx"
done
"$bin/policy_alignment_bundle_checker_test" "$validation_dir/bundle_checker"
export M20_ALIGNMENT_MODEL_SHA256="$(sha256sum "$release/policy/model.onnx" | cut -d' ' -f1)"
"$bin/policy_alignment_bundle_fixture" "$release/policy/model.onnx" "$validation_dir/bundle_runner.jsonl"
"$bin/policy_alignment_check" "$validation_dir/bundle_runner.jsonl" "$validation_dir/bundle_runner.paired.jsonl" "$release/policy/model.onnx"
"$bin/policy_alignment_fixture" "$release/policy/model.onnx" "$validation_dir/alignment.jsonl"
"$bin/policy_alignment_check" "$validation_dir/alignment.jsonl" "$validation_dir/alignment.paired.jsonl" "$release/policy/model.onnx"
bash "$tools/run_native_contract_test.sh" "$release"
if [[ -f "$release/joint_safety_limits.json" ]]; then
  "$bin/joint_command_guard_test"
  mkdir -m 700 "$validation_dir/incident_recorder" "$validation_dir/incident_io_failure" "$validation_dir/incident_freeze" "$validation_dir/incident_startup" "$validation_dir/dds_no_motor"
  "$bin/incident_recorder_test" "$validation_dir/incident_recorder"
  "$bin/incident_recorder_test" "$validation_dir/incident_io_failure" io_failure
  "$bin/incident_recorder_test" "$validation_dir/incident_freeze" freeze
  "$bin/incident_recorder_test" "$validation_dir/incident_startup" startup
  "$bin/dds_joint_guard_no_motor_test" "$validation_dir/dds_no_motor"
  python3 - "$validation_dir/dds_no_motor/alignment.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1])]
assert rows[-1]['event']=='capture_end' and rows[-1]['complete']
published=[r for r in rows if r.get('event')=='joint_command_published' and r.get('policy_guarded')]
assert published
assert all(r['guard_mono_s']<=r['publish_begin_mono_s']<=r['mono_s'] for r in published)
print('DDS_GUARD_TRACE=PASS samples='+str(len(published))+' guard_precedes_actual_publish')
PY
  for joint in 0 1 2 4 5 6 8 9 10 12 13 14; do
    mkdir -m 700 "$validation_dir/dds_joint_$joint"
    "$bin/dds_joint_guard_no_motor_test" "$validation_dir/dds_joint_$joint" --joint "$joint"
  done
  python3 - "$release" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);p=root/'AOS_RELEASE_MANIFEST.json';m=json.loads(p.read_text())
m.update(native_rebuild_required=False,offline_validation_complete=True,
         incident_startup_revision='startup_safety_wait_v1',
         joint_guard_elf_sha256=hashlib.sha256((root/'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy').read_bytes()).hexdigest(),
         ready_for_motor_control=False,hardware_locomotion_validation='not_performed')
m['aos_elf_sha256']=m['joint_guard_elf_sha256']
p.write_text(json.dumps(m,indent=2)+'\n')
PY
  python3 "$tools/release_utils.py" "$release"
else
  echo 'JOINT_SAFETY_TESTS=SKIPPED release_has_no_joint_guard'
fi
sha256sum -c --quiet RELEASE_SHA256SUMS
echo 'NATIVE_1516_BUILD=PASS scope=offline_no_joint_commands'
