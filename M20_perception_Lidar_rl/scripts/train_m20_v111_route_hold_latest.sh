#!/usr/bin/env bash
# Manual launch: load the latest saved actor/critic, fresh optimizer/objective.
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
export M20_PYTHON="${M20_PYTHON:-/home/arcus/miniconda3/envs/isaaclab/bin/python}"
export PYTHONPATH="$TRAIN_ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export M20_FLAT_START_REPAIR=0 M20_ENTRY_DIRECT_REPAIR=0
REFERENCE="$TRAIN_ROOT/artifacts/route_hold_20261125/reference56898"
export M20_RECOVERY_TEACHER="$REFERENCE/teacher/model_56898.pt"
export M20_RECOVERY_REPLAY="$TRAIN_ROOT/data/m20_recovery/route_hold_20261125/current_gait.npz"
export M20_ROUTE_FLAT_REPLAY="$TRAIN_ROOT/data/m20_recovery/route_hold_20261125/route_flat.npz"
export M20_COMPACT_REPLAY="$TRAIN_ROOT/data/m20_recovery/route_hold_20261125/compact_flat.npz"
export M20_RETENTION_RESULTS="$REFERENCE/native/results.json"
export M20_ENTRY_RESULTS="$REFERENCE/entry/results.json"
export M20_FLAT_RESULTS="$REFERENCE/flat_start/results.json"
export M20_FORWARD_RESULTS="$REFERENCE/production/results.json"
export M20_ROUTE_TRACTION_RESULTS="$TRAIN_ROOT/artifacts/route_traction_20261125/reference58098/traction_b1_b3/results.json"
export M20_TRACTION_TEACHER="$TRAIN_ROOT/logs/rsl_rl/M20-V11.1-RouteHold/2026-11-25_11-30-18_303338_route_hold_latest/model_58098.pt"
TASK='M20-Stair-V11.1-RouteHold-v0'
M20_TOTAL="${M20_TOTAL_UPDATES:-50000}"
M20_ENVS="${M20_NUM_ENVS:-2048}"
M20_EVALUATION="${M20_EVAL_EVERY:-1000}"
SESSION="${M20_SESSION_OUTPUT:-$TRAIN_ROOT/artifacts/route_hold_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
SOURCE_RUN="${M20_SOURCE_RUN:-$TRAIN_ROOT/logs/rsl_rl/M20-V11.1-FlatPosture}"
ROUTE_RUN="$TRAIN_ROOT/logs/rsl_rl/M20-V11.1-RouteHold"
INITIAL="${M20_INITIAL_CHECKPOINT:-}"
if [[ -z "$INITIAL" ]]; then
  INITIAL="$("$M20_PYTHON" - "$SOURCE_RUN" "$ROUTE_RUN" <<'PY'
from pathlib import Path
import sys
files = []
for source in sys.argv[1:]:
    run = Path(source)
    pattern = 'model_*.pt' if (run/'params/env.yaml').is_file() else '*/model_*.pt'
    files += [p for p in run.glob(pattern) if p.stem.removeprefix('model_').isdigit()]
if not files:
    raise SystemExit('No saved FlatPosture/RouteHold checkpoint')
print(max(files, key=lambda p: (p.stat().st_mtime_ns, int(p.stem.split('_')[-1]))).resolve())
PY
)"
fi
LOAD_FLAGS=(--initial-checkpoint "$INITIAL")
RESUME_FLAGS=()
if [[ -n "${M20_RESUME_CHECKPOINT:-}" ]]; then
  INITIAL="$M20_RESUME_CHECKPOINT"
  LOAD_FLAGS=(--resume-checkpoint "$INITIAL")
  RESUME_FLAGS=(--resume "$INITIAL")
fi
echo "[M20 RouteHold] initial checkpoint: $INITIAL"
echo "[M20 RouteHold] flat/stair preservation teacher: $M20_RECOVERY_TEACHER"
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py \
  --num_envs "$M20_ENVS" --max_iterations 100
"$M20_PYTHON" scripts/tools/m20_recovery_preflight.py \
  --checkpoint "$INITIAL" --teacher "$M20_RECOVERY_TEACHER" \
  --replay "$M20_RECOVERY_REPLAY" --targeted-contract
"$M20_PYTHON" scripts/tools/m20_joint_limits_preflight.py --checkpoint "$INITIAL"
"$M20_PYTHON" scripts/tools/m20_route_hold_preflight.py \
  --checkpoint "$INITIAL" --teacher "$M20_RECOVERY_TEACHER" \
  --replay "$M20_RECOVERY_REPLAY" --compact "$M20_COMPACT_REPLAY" \
  --native "$M20_RETENTION_RESULTS" --entry "$M20_ENTRY_RESULTS" \
  --flat "$M20_FLAT_RESULTS" --production "$M20_FORWARD_RESULTS" \
  --route-flat "$M20_ROUTE_FLAT_REPLAY" --traction-results "$M20_ROUTE_TRACTION_RESULTS" \
  --traction-teacher "$M20_TRACTION_TEACHER" --joint-limits "${RESUME_FLAGS[@]}"
"$M20_PYTHON" - "$TASK" "$M20_TOTAL" "$M20_EVALUATION" "$M20_RETENTION_RESULTS" "$SESSION" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/reinforcement_learning/rsl_rl').resolve()))
from guarded_recovery import validate_session, resource_check
limits = validate_session(sys.argv[1], int(sys.argv[2]), 100, sys.argv[4], sys.argv[5])
if int(sys.argv[3]) < 100 or int(sys.argv[3]) % 100:
    raise ValueError('M20_EVAL_EVERY must be a positive multiple of 100')
resource_check(Path.cwd(), limits)
print('[M20] Route hold objective validated; behavior failures record and continue.')
PY
if [[ "${1:-}" == '--check-only' ]]; then
  exit 0
fi
if (($#)); then
  echo "Unsupported argument: $1. Use M20_TOTAL_UPDATES/M20_NUM_ENVS/M20_EVAL_EVERY." >&2
  exit 2
fi
export M20_REPAIR_RUN_RECEIPT="$SESSION/run_dir.txt"
exec "$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" --headless --device cuda:0 --num_envs "$M20_ENVS" --max_iterations 100 --seed 42 \
  "${LOAD_FLAGS[@]}" --run_name route_hold_latest \
  --guarded-total-updates "$M20_TOTAL" --guarded-eval-every "$M20_EVALUATION" \
  --guarded-eval-profile entry --guard-baseline "$M20_RETENTION_RESULTS" --guard-output "$SESSION"
