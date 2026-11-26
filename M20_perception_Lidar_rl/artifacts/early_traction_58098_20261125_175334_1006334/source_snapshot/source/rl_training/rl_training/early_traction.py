"""58098 repair contracts and simulator-independent retention math."""
import hashlib
from pathlib import Path
import numpy as np
import torch

PROTOCOL = 'm20_early_handoff_58098_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_band_loss(prediction, target, early_wheel):
    """Retain full actions outside B1-B3; allow local sagittal corrections.

    Early wheels have no velocity imitation target. HipX remains protected;
    the corresponding HipY/Knee pair gets a .04rad (action .16) dead band.
    Later stairs, descent and flat use .02rad sagittal / 1rad/s wheel bands.
    """
    early_leg = early_wheel.repeat_interleave(3, dim=1)
    tolerance = prediction.new_tensor([.08]*12+[.20]*4).expand_as(prediction).clone()
    sagittal = torch.tensor([False, True, True]*4, device=prediction.device)
    tolerance[:, :12] = torch.where(early_leg & sagittal, .16, tolerance[:, :12])
    weights = torch.ones_like(prediction)
    weights[:, 12:] = (~early_wheel).to(prediction.dtype)
    weights[:, :12] = torch.where(early_leg & sagittal, .35, 1.)
    error = ((prediction-target.detach()).abs()-tolerance).clamp_min(0).square()
    return (error*weights).sum(1).div(weights.sum(1).clamp_min(1)).mean()


def descent_pose(trajectory):
    """Metrics from pre-control state, alongside actual executed actions."""
    with np.load(trajectory, allow_pickle=False) as d:
        active = (d['mode'] == 'policy') & (d['command'][:, 0] > .1)
        if active.sum() < 10:
            raise ValueError('Descent has insufficient policy samples')
        q = d['root_quat'][active]
        pitch = np.degrees(np.arcsin(np.clip(2*(q[:, 0]*q[:, 2]-q[:, 3]*q[:, 1]), -1, 1)))
        roll = np.degrees(np.arctan2(2*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3]),
                                    1-2*(q[:, 1]**2+q[:, 2]**2)))
        yaw = np.unwrap(np.arctan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                                  1-2*(q[:, 2]**2+q[:, 3]**2)))
        pos, t = d['root_pos'][active], d['time'][active]
        # Actual MuJoCo wheel-to-hip geometry, projected on the base axis;
        # never approximate reach from action targets or guessed link lengths.
        reach = d['wheel_reach'][active]
        return dict(pitch_peak_deg=float(np.abs(pitch).max()),
            roll_peak_deg=float(np.abs(roll).max()),
            reach_p95_m=float(np.percentile(np.abs(reach).max(1), 95)),
            yaw_peak_deg=float(np.degrees(np.abs(yaw-yaw[0])).max()),
            lateral_peak_m=float(np.abs(pos[:, 1]-pos[0, 1]).max()),
            mean_speed_m_s=float((pos[-1, 0]-pos[0, 0])/max(.02, t[-1]-t[0])),
            moving_seconds=float(t[-1]-t[0]), samples=int(active.sum()))
