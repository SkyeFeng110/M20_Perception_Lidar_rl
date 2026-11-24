#!/usr/bin/env bash
# Manual start: strict B1 transfer plus bounded flat full-stick correction.
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
export M20_PYTHON="${M20_PYTHON:-python3}"
export PYTHONPATH="$TRAIN_ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export M20_FLAT_START_REPAIR=0 M20_ENTRY_DIRECT_REPAIR=0
export M20_RECOVERY_TEACHER="${M20_RECOVERY_TEACHER:-$TRAIN_ROOT/interview_prep/model_49998.pt}"
export M20_RECOVERY_REPLAY="$TRAIN_ROOT/data/m20_recovery/reference_49998_sequence_20261124.npz"
REFERENCE="$TRAIN_ROOT/artifacts/entry_49998_repair_20261124/reference"
export M20_RETENTION_RESULTS="$REFERENCE/native/results.json"
export M20_ENTRY_RESULTS="$REFERENCE/entry/results.json"
export M20_FLAT_RESULTS="$REFERENCE/flat_start/results.json"
export M20_FORWARD_RESULTS="$REFERENCE/production/results.json"
TASK='M20-Stair-V11.1-EntrySequenceRepair-v0'
M20_TOTAL="${M20_TOTAL_UPDATES:-50000}"
M20_ENVS="${M20_NUM_ENVS:-2048}"
M20_EVALUATION="${M20_EVAL_EVERY:-1000}"
SESSION="${M20_SESSION_OUTPUT:-$TRAIN_ROOT/artifacts/sequence_from49998_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
LOAD_FLAGS=(--initial-checkpoint "$M20_RECOVERY_TEACHER")
RESUME_FLAGS=()
if [[ -n "${M20_RESUME_CHECKPOINT:-}" ]]; then
  LOAD_FLAGS=(--resume-checkpoint "$M20_RESUME_CHECKPOINT")
  RESUME_FLAGS=(--resume "$M20_RESUME_CHECKPOINT")
fi
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py \
  --num_envs "$M20_ENVS" --max_iterations 100
"$M20_PYTHON" scripts/tools/m20_recovery_preflight.py \
  --checkpoint "${M20_RESUME_CHECKPOINT:-$M20_RECOVERY_TEACHER}" \
  --teacher "$M20_RECOVERY_TEACHER" --replay "$M20_RECOVERY_REPLAY" --targeted-contract
"$M20_PYTHON" scripts/tools/m20_entry_sequence_preflight.py \
  --teacher "$M20_RECOVERY_TEACHER" --replay "$M20_RECOVERY_REPLAY" \
  --native "$M20_RETENTION_RESULTS" --entry "$M20_ENTRY_RESULTS" \
  --flat "$M20_FLAT_RESULTS" --production "$M20_FORWARD_RESULTS" "${RESUME_FLAGS[@]}"
"$M20_PYTHON" - "$TASK" "$M20_TOTAL" "$M20_EVALUATION" "$M20_RETENTION_RESULTS" "$SESSION" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/reinforcement_learning/rsl_rl').resolve()))
from guarded_recovery import validate_session, resource_check
limits = validate_session(sys.argv[1], int(sys.argv[2]), 100, sys.argv[4], sys.argv[5])
if int(sys.argv[3]) < 100 or int(sys.argv[3]) % 100:
    raise ValueError('M20_EVAL_EVERY must be a positive multiple of 100')
resource_check(Path.cwd(), limits)
print('[M20] Seed/evidence/resources validated. Behavioral failures record and continue.')
PY
if [[ "${1:-}" == '--check-only' ]]; then
  exit 0
fi
if [[ "${1:-}" == '--probe-only' ]]; then
  shift
  exec "$M20_PYTHON" scripts/tools/probe_m20_entry_sequence.py \
    --checkpoint "${M20_RESUME_CHECKPOINT:-$M20_RECOVERY_TEACHER}" --headless --device cuda:0 \
    --output "$TRAIN_ROOT/artifacts/entry_49998_repair_20261124/frozen_runtime.json" "$@"
fi
if (($#)); then
  echo "Unsupported argument: $1. Configure M20_TOTAL_UPDATES, M20_NUM_ENVS or M20_EVAL_EVERY." >&2
  exit 2
fi
export M20_REPAIR_RUN_RECEIPT="$SESSION/run_dir.txt"
exec "$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" --headless --device cuda:0 --num_envs "$M20_ENVS" --max_iterations 100 --seed 42 \
  "${LOAD_FLAGS[@]}" --run_name sequence_flat_from49998 \
  --guarded-total-updates "$M20_TOTAL" --guarded-eval-every "$M20_EVALUATION" \
  --guarded-eval-profile entry --guard-baseline "$M20_RETENTION_RESULTS" --guard-output "$SESSION"
