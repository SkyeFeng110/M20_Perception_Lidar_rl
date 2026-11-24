"""Finite next-tread credit; simulation reward math, never actor observations."""
import math
import torch


def support_credit_step(s, expected, x, bottom, velocity_z, force_z, lower,
                        rise, tread, in_course, active, reset, dt):
    """Pay a finite ledger per front wheel/target, reserving most for dwell.

    A lift can earn .05, a descent .15, one/two/three consecutive loaded
    frames .43/.72/1 at 50 Hz. Oscillating, holding, command toggles and
    backtracking cannot refill a target. A missed target does not disable
    later targets: the physical support ledger resynchronizes after a skip.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Support credit requires a positive finite policy dt')
    margin = torch.minimum(torch.full_like(tread, .035), tread*.15)[:, None]
    start = (expected-1)*tread[:, None]+margin
    end = expected*tread[:, None]-margin
    plane = lower[:, None]+expected*rise[:, None]
    error = bottom-plane
    outside = torch.maximum(start-x, x-end).clamp_min(0)
    alignment = torch.exp(-(outside/.10).square())*in_course*(x <= end)
    near = torch.exp(-(error/.08).square())*(error >= -.035)
    descent = ((-velocity_z-.02)/.20).clamp(0, 1)
    hint = alignment*near*(.05+.10*descent)
    loaded = (in_course & (x >= start) & (x <= end)
              & (error.abs() <= .025) & (force_z >= 15.))
    same = expected == s['target']
    # Consecutive frames must be at the same physical target, even while the
    # command is closed; closed commands consume credit but never pay it.
    frames = torch.where(loaded & ~reset[:, None],
                         torch.where(same, s['frames']+1, 1), 0)
    dwell = (frames/max(1, math.ceil(.06/dt))).clamp(0, 1)
    score = torch.maximum(hint, (.15+.85*dwell)*loaded)
    prior = torch.where(same, s['peak'], torch.zeros_like(score))
    prior = torch.where(reset[:, None], score, prior)
    forward_target = (expected >= s['target']) | reset[:, None]
    credit = (score-prior).clamp_min(0)*active*~reset[:, None]*forward_target
    s.update(target=torch.where(forward_target, expected, s['target']),
             peak=torch.where(forward_target, torch.maximum(prior, score), s['peak']),
             frames=frames)
    return credit
