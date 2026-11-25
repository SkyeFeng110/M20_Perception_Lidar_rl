"""Resolve M20 training target bounds from the real deployment's named limits."""
import json
import math
from pathlib import Path

LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("fl", "fr", "hl", "hr")
    for joint in ("hipx", "hipy", "knee")
)


def joint_target_contract(limits_root=None):
    root = (Path(limits_root) if limits_root is not None else
            Path(__file__).resolve().parents[4] / "mujoco_validation/deploy_real")
    safety = json.loads((root / "joint_safety_limits.json").read_text())
    target = json.loads((root / "policy_target_limits.json").read_text())
    lower, upper, margin = (safety.get("user_lower_rad", ()), safety.get("user_upper_rad", ()),
                            target.get("margin_rad", ()))
    if (safety.get("schema") != 1 or target.get("schema") != 1
            or not safety.get("operator_limits_confirmed")
            or len(lower) != 12 or len(upper) != 12 or len(margin) != 3):
        raise ValueError("The confirmed M20 leg joint safety limits are incomplete")
    if (safety.get("reference_robot_sha256") != target.get("reference_robot_sha256")
            or not all(math.isfinite(float(x)) for x in (*lower, *upper, *margin))
            or any(float(x) < 0 for x in margin)):
        raise ValueError("M20 joint target bounds and safety margins have inconsistent sources or values")
    bounds = [[round(float(lo)+float(pad), 9), round(float(hi)-float(pad), 9)]
              for lo, hi, pad in zip(lower, upper, list(margin)*4)]
    if not all(lo < hi for lo, hi in bounds):
        raise ValueError("The M20 leg target bounds are invalid")
    return dict(protocol="m20_named_leg_target_limits_v1", joint_names=list(LEG_JOINT_NAMES),
                bounds_rad=bounds, margin_rad=list(margin),
                action_history="executed_last_action", excess_cost_weight=-0.1)


def joint_target_bounds():
    return tuple(tuple(pair) for pair in joint_target_contract()["bounds_rad"])


def joint_target_clip():
    return dict(zip(LEG_JOINT_NAMES, joint_target_bounds()))
