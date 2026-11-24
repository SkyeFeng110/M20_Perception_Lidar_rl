#!/usr/bin/env bash
# Native offline test only: never instantiates a ROS/SDK motor publisher.
set -euo pipefail
release="${1:?Usage: run_native_contract_test.sh AOS_RELEASE}"
release="$(cd "$release" && pwd)"
tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -s "$tools/native_fixture.txt" ]]; then
  echo 'NATIVE_CONTRACT_REPLAY=FAIL missing native_fixture.txt; generate with make_native_fixture.py for the deployed checkpoint' >&2
  exit 2
fi
src="$release/src/M20_sdk_deploy"
mkdir -p "$release/artifacts"
nice -n 19 g++ -std=c++17 -O2 \
  -I "$src/run_policy" -I "$src/include/types" -I "$src/include/utils" \
  -I "$src/third_party/eigen" -I "$src/third_party/onnxruntime/arm/include" \
  "$tools/native_contract_replay.cpp" \
  -L "$src/third_party/onnxruntime/arm/lib" \
  -Wl,-rpath,"$src/third_party/onnxruntime/arm/lib" \
  -lonnxruntime -lpthread -ldl -o "$release/artifacts/native_contract_replay"
export M20_V11_FEATURE_CONTRACT=v11_signed_edges_v2
"$release/artifacts/native_contract_replay" "$release/policy/model.onnx" "$tools/native_fixture.txt"
