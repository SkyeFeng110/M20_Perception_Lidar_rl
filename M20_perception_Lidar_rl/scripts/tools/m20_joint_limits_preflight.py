"""Check deployment-derived target bounds against saved M20 physical trajectories."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training/rl_training"))
sys.path.insert(0, str(ROOT.parent / "mujoco_validation/deploy/mujoco_v135"))

from joint_target_limits import joint_target_contract
from policy import SavedPolicy


def validate(checkpoint):
    contract = joint_target_contract()
    policy = SavedPolicy(checkpoint)
    lower = np.asarray(contract["bounds_rad"], dtype=np.float64)[:, 0]
    upper = np.asarray(contract["bounds_rad"], dtype=np.float64)[:, 1]
    roots = (ROOT / "artifacts/entry_49998_repair_20261124/reference",
             ROOT / "artifacts/flat_posture_repair_20261124/reference50498")
    trajectories = []
    for root in roots:
        trajectories.extend(sorted(root.rglob("*.npz")))
    checked, samples, violations, max_excess = 0, 0, 0, 0.0
    for path in trajectories:
        with np.load(path, allow_pickle=False) as data:
            if "action" not in data or data["action"].ndim != 2 or data["action"].shape[1] != 16:
                continue
            action = data["action"][:, :12]
            if not np.isfinite(action).all():
                raise ValueError(f"Nonfinite saved action: {path}")
            target = policy.nominal[:12] + policy.action_scale[:12] * action
            excess = np.maximum(np.maximum(lower - target, target - upper), 0.0)
            violations += int((excess.max(axis=1) > 1e-6).sum())
            max_excess = max(max_excess, float(excess.max(initial=0.0)))
            samples += len(action)
            checked += 1
    if checked < 50 or samples < 10000:
        raise ValueError(f"Insufficient measured baseline coverage: {checked} trajectories, {samples} samples")
    if violations:
        raise ValueError(f"Target bounds would alter accepted baseline actions: {violations} frames")
    return dict(status="joint_limits_preflight_passed", checkpoint=str(Path(checkpoint).resolve()),
                contract=contract, trajectory_files=checked, samples=samples,
                baseline_frames_clipped=violations, max_baseline_excess_rad=max_excess,
                no_training_started=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().checkpoint), indent=2))
