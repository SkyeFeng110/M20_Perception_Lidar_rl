#!/usr/bin/env bash
# Bootstrap the native supervisor; hardware controller requires its independent
# incident recorder even when optional alignment diagnostics are disabled.
set -euo pipefail
release="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:?Usage: launch_aos.sh aos-preflight|aos-run EXPECTED_MAC}"
expected_mac="${2:?Pass the independently verified target AOS MAC}"
case "$mode" in aos-preflight|aos-run) ;; *) exit 2;; esac
export LD_LIBRARY_PATH="$release/src/M20_sdk_deploy/third_party/onnxruntime/arm/lib:${LD_LIBRARY_PATH:-}"
if [[ "$mode" == aos-run ]]; then
  [[ -f "$release/joint_safety_limits.json" ]]
  [[ -f "$release/src/M20_sdk_deploy/run_policy/joint_command_guard.hpp" ]]
  python3 - "$release" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'AOS_RELEASE_MANIFEST.json').read_text())
binary=root/'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy'
if (manifest.get('joint_safety_revision')!='joint_publish_guard_v1' or
    manifest.get('native_rebuild_required') is not False or
    manifest.get('offline_validation_complete') is not True or
    hashlib.sha256(binary.read_bytes()).hexdigest()!=manifest.get('joint_guard_elf_sha256')):
    raise SystemExit('Refusing motor control: joint guard rebuild/validation/ELF identity missing')
PY
  # Persistent robot data partition, independent of the SSH session and release
  # directory. The fixed ring is reused across controller sessions.
  incident_dir=/var/opt/robot/data/m20_joint_guard_incidents
  install -d -m 700 "$incident_dir"
  export M20_INCIDENT_DIR="$incident_dir"
  export M20_INCIDENT_MODEL_SHA256="$(sha256sum "$release/policy/model.onnx" | cut -d' ' -f1)"
fi
exec "$release/native_tools/m20_deploy" "$mode" --release "$release" \
  --probe "$release/runtime_probe/m20_deployment_probe" \
  --expected-mac "$expected_mac" --diagnostics off
