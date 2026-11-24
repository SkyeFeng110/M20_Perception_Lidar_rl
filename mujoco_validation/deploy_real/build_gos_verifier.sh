#!/usr/bin/env bash
# Restore the saved runtime's missing offline acceptance executable. No motors.
set -euo pipefail
release="$(cd "${1:?RELEASE EXPECTED_MAC EXPECTED_SERIAL}" && pwd)"
expected_mac="${2:?EXPECTED_MAC}"
expected_serial="${3:?EXPECTED_SERIAL}"
[[ "$EUID" == 0 && "$(uname -m)" == aarch64 ]]
[[ "$(tr -d '\0' </sys/firmware/devicetree/base/serial-number)" == "$expected_serial" ]]
python3 - "$expected_mac" <<'PY'
from pathlib import Path
import sys
assert sys.argv[1].lower() in {p.read_text().strip().lower() for p in Path('/sys/class/net').glob('*/address')}
PY
if pgrep -x m20_deploy >/dev/null; then
  echo 'Stop the native GOS supervisor before installing its verifier' >&2
  exit 1
fi
cd "$release"
sha256sum -c --quiet RELEASE_SHA256SUMS
# Reference ARM compiler limit: 4 GiB plus 1 GiB reserve, one compiler at a time.
available_kib="$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)"
(( available_kib > 5*1024*1024 ))
(( $(df -Pk "$release" | awk 'NR==2 {print $4}') > 512*1024 ))
free -m
source_root="$release/runtime/source"
cpp="$source_root/tools/gos_stairdetector_6d/cpp"
sdk="$source_root/src/M20_sdk_deploy"
out="$(mktemp -d "$release/artifacts/verifier_build_XXXXXXXX")"
for target in verify_run verify_run_test; do
  nice -n 19 taskset -c 4,5 prlimit --as=4294967296 -- \
    g++ -std=c++17 -O2 -pthread -I /usr/include/eigen3 -I "$sdk/include/utils" -I "$sdk/run_policy" -I "$cpp" \
    "$cpp/$target.cpp" -o "$out/$target"
done
"$out/verify_run_test" "$out/verify_run"
cp "$out/verify_run" "$release/runtime/build/gos6d_verify_run.next"
chmod 755 "$release/runtime/build/gos6d_verify_run.next"
mv "$release/runtime/build/gos6d_verify_run.next" "$release/runtime/build/gos6d_verify_run"
python3 "$(dirname "$0")/release_utils.py" "$release"
sha256sum "$release/runtime/build/gos6d_verify_run" >"$out/installed_sha256.txt"
echo 'GOS_VERIFIER_BUILD=PASS no_motor_io'
