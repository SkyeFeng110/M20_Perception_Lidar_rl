"""Bounded flat-only acceleration repair. No new actor inputs or stair clock."""
import importlib.util
from pathlib import Path
import math
import torch

_path = Path(__file__).resolve().parents[4]/'mujoco_validation/deploy/mujoco_v135/command_filter.py'
_spec = importlib.util.spec_from_file_location('m20_forward_contract', _path)
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)


def filter_forward(current, requested, dt):
    return _contract.forward_slew(current, requested, dt, where=torch.where, clip=torch.clamp)


def direct_entry_stress_mask(indices, terrain_types, flat_columns, ascent_columns,
                             enabled=False):
    """Select raw command cohorts for the current robot's direct-stick path."""
    indices = torch.as_tensor(indices, device=terrain_types.device, dtype=torch.long)
    flat = flat_columns[terrain_types]
    ascent = ascent_columns[terrain_types]
    if enabled:
        # Half of ascent samples practice the raw first-riser transition;
        # one quarter of flat samples keeps the deployment posture pressure.
        return (ascent & (indices.remainder(2) == 0)) | (flat & (indices.remainder(4) == 0))
    return (flat | ascent) & (indices.remainder(8) == 3)


def flat_start_terms(wheel_y, hipx, pitch, pitch_rate, action, previous_action,
                     command, previous_command, window, active, reset, dt):
    """Physical body-frame geometry and finite transient cost budget per second.

    `active` is privileged true flat-column evidence, never actor edge absence.
    No stance pattern or left/right lead is prescribed. The entire cost is
    bounded by .35; velocity/progress rewards are independent of this cost.
    """
    previous_command = torch.where(reset, command, previous_command)
    # Includes partial->full and ramped acceleration, not only starts at zero.
    accelerating = (command-previous_command > .005) & (command > .08) & active & ~reset
    window = torch.where(reset | ~active, 0., (window-dt).clamp_min(0))
    window = torch.where(accelerating, torch.full_like(window, .6), window)
    transient = (window > 0) & active & (command > .08) & ~reset
    sides = wheel_y.new_tensor([1., -1., 1., -1.])
    outward = wheel_y*sides
    # M20's nominal half stance is ~.26m. A .18m inner envelope provides
    # freedom for normal balance, while rejecting the reproduced 16cm tuck.
    inward = ((.18-outward).clamp_min(0)/.08).square().mean(1)
    front_width = wheel_y[:, 0]-wheel_y[:, 1]
    narrow = ((.36-front_width).clamp_min(0)/.12).square()
    hip = ((hipx.abs()-math.radians(10)).clamp_min(0)/math.radians(10)).square().mean(1)
    lateral = (inward+.5*narrow+.25*hip)
    lateral = lateral/(1+lateral)
    tilt = ((pitch.abs()-math.radians(3)).clamp_min(0)/math.radians(5)).square()
    rocking = ((pitch_rate.abs()-.25).clamp_min(0)/.75).square()
    attitude = .4*tilt/(1+tilt)+.6*rocking/(1+rocking)
    # Judge target changes in physical units, including hipx. No wheel sign
    # or absolute drive magnitude is imposed; useful rolling is not penalized.
    delta = action-previous_action
    scale = action.new_tensor([.125, .25, .25]*4+[5.]*4)
    free = action.new_tensor([math.radians(4), math.radians(10), math.radians(10)]*4+[2.5]*4)
    impulse = ((delta.abs()*scale-free).clamp_min(0)/free).square().mean(1)
    impulse = impulse/(1+impulse)
    moving = active & (command > .08) & ~reset
    cost = (.15*lateral+.12*attitude)*moving+.08*impulse*transient
    return dict(cost=cost, lateral_cost=lateral, attitude_cost=attitude,
                impulse_cost=impulse, window=window, transient=transient,
                front_width=front_width, min_outward=outward.amin(1),
                hipx_abs=hipx.abs().amax(1), pitch_abs=pitch.abs(),
                pitch_rate_abs=pitch_rate.abs(), active=moving)
