#!/usr/bin/env bash
# Bounded sensor/observation run. No robot motor controller is executed.
set -euo pipefail
release="$(cd "${1:?GOS_RELEASE EXPECTED_MAC OUTPUT}" && pwd)"
expected_mac="${2:?EXPECTED_MAC}"
output="${3:?Unique OUTPUT path}"
[[ "$EUID" == 0 ]]
cd "$release"
sha256sum -c --quiet RELEASE_SHA256SUMS
exec "$release/native_tools/m20_deploy" gos-run \
  --runtime "$release/runtime" --stage "$release/stage" --output "$output" \
  --expected-mac "$expected_mac" --seconds 20 --diagnostics off --acceptance off
