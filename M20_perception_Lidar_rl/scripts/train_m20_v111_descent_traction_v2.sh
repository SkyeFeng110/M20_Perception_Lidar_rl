#!/usr/bin/env bash
# Explicit V2: latest initial policy, 58098 retention teacher, fresh optimizer.
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
export M20_PYTHON="${M20_PYTHON:-/home/mark/miniconda3/envs/isaaclab-5.1/bin/python}"
if [[ "${1:-}" != '--check-only' ]]; then
  "$M20_PYTHON" - <<'PY'
from pathlib import Path
import os
running = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().decode().strip('\0').split('\0')
    except (OSError, UnicodeError):
        continue
    if (int(path.parent.name) != os.getpid() and any(Path(a).name == 'train.py' for a in args)
            and '--task' in args and args[args.index('--task')+1] == 'M20-Stair-V11.1-RouteHold-v0'
            and '--device' in args and args[args.index('--device')+1] == 'cuda:0'):
        running.append(path.parent.name)
if running:
    raise SystemExit('Existing M20 GPU training PID(s): '+', '.join(running)+
                     '. Stop the old run with Ctrl+C in its terminal before launching V2.')
PY
fi
export M20_DESCENT_TRACTION_V2=1
export M20_INITIAL_CHECKPOINT="${M20_INITIAL_CHECKPOINT:-$TRAIN_ROOT/logs/rsl_rl/M20-V11.1-EarlyTraction58098/2026-11-25_17-53-45_450637_early_traction_from58098/model_65298.pt}"
export M20_TOTAL_UPDATES="${M20_TOTAL_UPDATES:-30000}"
checkpoint_tag="${M20_INITIAL_CHECKPOINT##*/}"
checkpoint_tag="${checkpoint_tag%.pt}"
export M20_RUN_NAME="${M20_RUN_NAME:-descent_traction_front_turn_v2_from${checkpoint_tag#model_}}"
exec bash scripts/train_m20_v111_early_traction_from58098.sh "$@"
