#!/usr/bin/env bash
# User-started fine-tuning. Baseline capture is cached; PPO uses a fresh optimizer.
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
CHECK_ONLY=0
if [[ "${1:-}" == '--check-only' && $# == 1 ]]; then
  CHECK_ONLY=1
elif (($#)); then
  echo 'Use M20_TOTAL_UPDATES, M20_NUM_ENVS, M20_EVAL_EVERY and M20_RESUME_CHECKPOINT.' >&2
  exit 2
fi
export M20_PYTHON="${M20_PYTHON:-/home/mark/miniconda3/envs/isaaclab-5.1/bin/python}"
export PYTHONPATH="$TRAIN_ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export M20_FLAT_START_REPAIR=0 M20_ENTRY_DIRECT_REPAIR=0 M20_FLAT_STABILITY_REPAIR=0
export M20_EARLY_TRACTION_58098=1
BASE="$TRAIN_ROOT/artifacts/early_traction_58098_20261125/reference58098/teacher/model_58098.pt"
INITIAL="${M20_INITIAL_CHECKPOINT:-$BASE}"
LOAD_FLAGS=(--initial-checkpoint "$INITIAL")
RESUME_FLAGS=()
if [[ -n "${M20_RESUME_CHECKPOINT:-}" ]]; then
  INITIAL="$M20_RESUME_CHECKPOINT"
  LOAD_FLAGS=(--resume-checkpoint "$INITIAL")
  RESUME_FLAGS=(--resume "$INITIAL")
fi
# Fail before asset preparation; never replace an explicitly requested model.
if [[ ! -f "$INITIAL" ]]; then
  "$M20_PYTHON" - "$INITIAL" <<'PY'
from pathlib import Path
import sys
requested = Path(sys.argv[1])
print('[M20] Checkpoint does not exist: '+str(requested), file=sys.stderr)
print('Unevaluated intermediate checkpoints may have been pruned by the previous run.', file=sys.stderr)
available = [p for p in requested.parent.glob('model_*.pt') if p.stem[6:].isdigit()]
for candidate in sorted(available, key=lambda p:int(p.stem[6:]), reverse=True)[:5]:
    print('Available: '+str(candidate), file=sys.stderr)
print('Set M20_INITIAL_CHECKPOINT to the intended existing checkpoint; no fallback was loaded.', file=sys.stderr)
PY
  exit 2
fi
REFERENCE="${M20_EARLY_REFERENCE:-$TRAIN_ROOT/artifacts/early_traction_58098_20261125/reference58098}"
DATA="${M20_EARLY_DATA:-$TRAIN_ROOT/data/m20_recovery/early_traction_58098_20261125}"
TASK='M20-Stair-V11.1-RouteHold-v0'
TOTAL="${M20_TOTAL_UPDATES:-1000}"
ENVS="${M20_NUM_ENVS:-2048}"
EVALUATION="${M20_EVAL_EVERY:-500}"
SESSION="${M20_SESSION_OUTPUT:-$TRAIN_ROOT/artifacts/early_traction_58098_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
if [[ "${M20_DESCENT_TRACTION_V2:-0}" == 1 ]]; then
  SESSION="${M20_SESSION_OUTPUT:-$TRAIN_ROOT/artifacts/descent_traction_v2_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
fi
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py \
  --num_envs "$ENVS" --max_iterations 100
"$M20_PYTHON" - "$TOTAL" "$EVALUATION" <<'PY'
import sys
total, evaluation = map(int, sys.argv[1:])
if not 100 <= total <= 50000 or total % 100 or evaluation < 100 or evaluation % 100:
    raise ValueError('TOTAL must be 100..50000; TOTAL and EVAL_EVERY must be multiples of 100')
PY
"$M20_PYTHON" scripts/tools/m20_prepare_early_traction.py \
  --checkpoint "$BASE" --reference "$REFERENCE" --data "$DATA"
export M20_RECOVERY_TEACHER="$REFERENCE/teacher/model_58098.pt"
export M20_RECOVERY_REPLAY="$DATA/current_gait.npz"
export M20_COMPACT_REPLAY="$DATA/compact_flat.npz"
export M20_ROUTE_FLAT_REPLAY="$DATA/route_flat.npz"
export M20_EARLY_PROTECTION_REPLAY="$DATA/protected_58098.npz"
export M20_RETENTION_RESULTS="$REFERENCE/native/results.json"
export M20_ENTRY_RESULTS="$REFERENCE/entry/results.json"
export M20_FLAT_RESULTS="$REFERENCE/flat_start/results.json"
export M20_FORWARD_RESULTS="$REFERENCE/production/results.json"
export M20_DESCENT_RESULTS="$REFERENCE/descent/results.json"
export M20_ROUTE_TRACTION_RESULTS="$TRAIN_ROOT/artifacts/route_traction_20261125/reference58098/traction_b1_b3/results.json"
export M20_TRACTION_TEACHER="$M20_RECOVERY_TEACHER"
if [[ "${M20_DESCENT_TRACTION_V2:-0}" == 1 ]]; then
  export M20_V2_REPLAY="${M20_V2_REPLAY:-$TRAIN_ROOT/data/m20_recovery/descent_traction_v2_20261126/protected_58098.npz}"
  export M20_FRONT_TURN_RESULTS="${M20_FRONT_TURN_RESULTS:-$TRAIN_ROOT/artifacts/descent_traction_v2_reference58098/front_turn/results.json}"
  "$M20_PYTHON" scripts/tools/m20_build_descent_traction_replay.py \
    --source "$M20_EARLY_PROTECTION_REPLAY" --teacher "$M20_RECOVERY_TEACHER" --output "$M20_V2_REPLAY"
fi
# Exercise the same resolver as the runner's parent config before Isaac starts.
"$M20_PYTHON" source/rl_training/rl_training/recovery_assets.py
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
"$M20_PYTHON" - "$TASK" "$TOTAL" "$EVALUATION" "$ENVS" "$SESSION" "$INITIAL" <<'PY'
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path[:0] = ['scripts/reinforcement_learning/rsl_rl', 'source/rl_training/rl_training']
from guarded_recovery import validate_session, resource_check
from early_traction_selection import validated_descent
from early_traction import PROTOCOL, sha
v2 = os.environ.get('M20_DESCENT_TRACTION_V2') == '1'
run_protocol = PROTOCOL
if v2:
    from descent_traction import PROTOCOL as run_protocol
    from front_turn import PROTOCOL as turn_protocol
task, total, evaluation, envs, session, initial = sys.argv[1:]
limits = validate_session(task, int(total), 100, os.environ['M20_RETENTION_RESULTS'], session)
resource_check(Path.cwd(), limits)
validated_descent(os.environ['M20_DESCENT_RESULTS'], os.environ['M20_RECOVERY_TEACHER'])
if os.environ.get('M20_RESUME_CHECKPOINT'):
    import torch
    expected = dict(protocol=PROTOCOL, protection_sha256=sha(os.environ['M20_EARLY_PROTECTION_REPLAY']),
        descent_reference_sha256=sha(os.environ['M20_DESCENT_RESULTS']),
        teacher_sha256=sha(os.environ['M20_RECOVERY_TEACHER']))
    if torch.load(initial, map_location='cpu', weights_only=True).get('m20_early_handoff_58098') != expected:
        raise ValueError('Resume requires this exact handoff objective, teacher and protection replay')
    if v2:
        expected_v2 = dict(protocol=run_protocol, repair_replay_sha256=sha(os.environ['M20_V2_REPLAY']),
                           teacher_sha256=sha(os.environ['M20_RECOVERY_TEACHER']), turn_protocol=turn_protocol)
        if torch.load(initial, map_location='cpu', weights_only=True).get('m20_descent_traction_v2', {}).get('contract') != expected_v2:
            raise ValueError('V2 needs initial-checkpoint or a matching V2 optimizer checkpoint')
output = Path(session)
output.mkdir(parents=True, exist_ok=True)
source = output/'source_snapshot'
paths = [Path('source/rl_training/rl_training/early_traction.py'),
    Path('source/rl_training/rl_training/v111_early_traction_runner.py'),
    Path('scripts/tools/m20_prepare_early_traction.py'),
    Path('scripts/reinforcement_learning/rsl_rl/early_traction_selection.py'),
    Path('scripts/train_m20_v111_early_traction_from58098.sh')]
if v2:
    paths += [Path('source/rl_training/rl_training/descent_traction.py'),
        Path('source/rl_training/rl_training/v111_descent_traction_runner.py'),
        Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v111_descent_traction.py'),
        Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v111_friction_cohorts.py'),
        Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v111_descent_traction_cfg.py'),
        Path('scripts/tools/m20_build_descent_traction_replay.py'),
        Path('scripts/reinforcement_learning/rsl_rl/descent_traction_selection.py'),
        Path('scripts/reinforcement_learning/rsl_rl/front_turn_selection.py'),
        Path('source/rl_training/rl_training/front_turn.py'),
        Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v111_front_turn.py'),
        Path('scripts/tools/m20_front_turn_benchmark.py'),
        Path('docs/M20_DESCENT_TRACTION_TURN_V2_20261126.md'),
        Path('scripts/train_m20_v111_descent_traction_v2.sh')]
for path in paths:
    dest = source/path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
diff = subprocess.check_output(['git', 'diff', '--binary'])
(output/'source_changes.patch').write_bytes(diff)
(output/'launch.json').write_text(json.dumps(dict(protocol=run_protocol,
    task=task, additional_updates=int(total), eval_every=int(evaluation), num_envs=int(envs),
    initial_checkpoint=initial, initial_sha256=sha(initial),
    teacher=os.environ['M20_RECOVERY_TEACHER'], teacher_sha256=sha(os.environ['M20_RECOVERY_TEACHER']),
    protection_sha256=sha(os.environ['M20_EARLY_PROTECTION_REPLAY']),
    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    source_files={str(p):sha(p) for p in paths}, learning_rate=1.5e-6 if v2 else 3.e-6,
    turn_protocol=turn_protocol if v2 else None,
    turn_reference_results=os.environ.get('M20_FRONT_TURN_RESULTS'),
    optimizer='resume' if os.environ.get('M20_RESUME_CHECKPOINT') else 'fresh'), indent=2)+'\n')
PY
if [[ "$CHECK_ONLY" == 1 ]]; then
  echo '[M20] CPU asset/config preflight passed; no PPO or Isaac simulation started.'
  exit 0
fi
if [[ "${M20_DESCENT_TRACTION_V2:-0}" == 1 ]]; then
  # Cache the new fixed turn baseline once when the user starts the job.
  # This block is deliberately skipped by the CPU-only preflight above.
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$M20_PYTHON" scripts/tools/m20_front_turn_benchmark.py \
      --checkpoint "$M20_RECOVERY_TEACHER" --output "$(dirname -- "$M20_FRONT_TURN_RESULTS")"
fi
echo "[M20 EarlyTraction58098] initial: $INITIAL"
echo "[M20 EarlyTraction58098] session: $SESSION"
export M20_REPAIR_RUN_RECEIPT="$SESSION/run_dir.txt"
exec "$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" --headless --device cuda:0 --num_envs "$ENVS" --max_iterations 100 --seed 42 \
  "${LOAD_FLAGS[@]}" --run_name "${M20_RUN_NAME:-early_traction_from58098}" \
  --guarded-total-updates "$TOTAL" --guarded-eval-every "$EVALUATION" \
  --guarded-eval-profile entry --guard-baseline "$M20_RETENTION_RESULTS" --guard-output "$SESSION"
