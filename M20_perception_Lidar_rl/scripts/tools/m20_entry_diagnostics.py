"""Physical first-riser diagnostics. No action or observation modification."""
import numpy as np

PROTOCOL = 'entry_transfer_v3'


def support_times(t, labels):
    """First sustained load per wheel/tread; phase times, not a fixed gait."""
    result = {}
    for index in sorted(int(v) for v in np.unique(labels) if v >= 1):
        times = []
        for side in range(4):
            times.append(next((float(t[i]) for i in range(len(t)-2)
                if (labels[i:i+3, side] == index).all()), None))
        result[str(index)] = times
    return result


def entry_report(arrays, *, radius, rise, tread, start_x, motion_start):
    t = np.asarray(arrays['time'])
    pos = np.asarray(arrays['wheel_position'])[:, :2]
    label = np.asarray(arrays['supported_tread'])[:, :2]
    root = np.asarray(arrays['position'])
    q = np.asarray(arrays['quaternion'])
    bottom = pos[:, :, 2] - radius
    moving = t >= motion_start
    lower = (label == 0) & (np.abs(bottom) < .035)
    # On a higher horizontal surface is supported, even though it is above
    # the lower ground. Height alone is NOT an airborne detector.
    lifted = (bottom > .035) & (label < 0)
    b1 = (label == 1) & (np.abs(bottom-rise) < .035)
    approached = (moving[:, None] & lower & (pos[:, :, 0] < start_x)).any(0)
    initial = int(np.searchsorted(t, motion_start).clip(0, len(t)-1))
    from_lower = bool(approached.all() and (pos[initial, :, 0] < start_x-radius).all()
                      and (bottom[initial] < .035).all())
    near = moving & (pos[:, :, 0].max(1) >= start_x-1.30)
    begin = int(np.flatnonzero(near)[0]) if near.any() else len(t)-1
    first_lift, first_support, confirmed = [], [], []
    for side in range(2):
        candidates = np.flatnonzero(lifted[:, side] & (np.arange(len(t)) >= begin))
        first_lift.append(int(candidates[0]) if len(candidates) else None)
        onset = ready = None
        for i in range(begin, len(t)-2):
            if b1[i:i+3, side].all():
                onset, ready = i, i+2
                break
        first_support.append(onset)
        confirmed.append(ready)
    valid_support = [i for i in first_support if i is not None]
    end = max(confirmed) if all(i is not None for i in confirmed) else min(
        len(t)-1, int(np.searchsorted(t, t[begin]+4.)))
    end = max(begin, end)
    scope = np.arange(begin, end+1)
    lead_side = int(np.argmin([i if i is not None else len(t) for i in first_support]))
    mate = 1-lead_side
    lead_ready = confirmed[lead_side]
    mate_lift = first_lift[mate]
    retained_support = bool(lead_ready is not None and confirmed[mate] is not None
        and b1[lead_ready:confirmed[mate]+1, lead_side].all())
    # The first B1 wheel must retain its load while the second wheel lifts.
    # Also require a real lift on each side; sliding a wheel up a riser is
    # not the alternating leg swing requested here.
    ordered = bool(all(i is not None for i in first_lift+confirmed)
        and first_lift[lead_side] < first_support[lead_side]
        and mate_lift > lead_ready and b1[mate_lift, lead_side]
        and not lifted[begin:lead_ready+1, mate].any()
        and lower[begin:lead_ready+1, mate].mean() >= .7 and retained_support)
    twin_seconds = float(lifted[scope].all(1).sum() * np.median(np.diff(t)))
    pitch = np.degrees(np.arcsin(np.clip(2*(q[:, 0]*q[:, 2]-q[:, 3]*q[:, 1]), -1, 1)))
    rate = np.degrees(np.asarray(arrays['angular_velocity'])[:, 1])
    root_vx = np.gradient(root[:, 0], t)
    wheel_vx = np.gradient(pos[:, :, 0], t, axis=0)
    # Includes approach compression, which can occur well before the wheels
    # physically reach the vertical riser. Loaded wheels are not necessarily
    # blocked by a collision: the leg policy can retract them under the body.
    blocked = (near & lower.all(1) & (root_vx-wheel_vx.max(1) > .15) & (root_vx > .25))
    dt = np.r_[np.diff(t), np.median(np.diff(t))]
    blocked_distance = float((np.maximum(root_vx-wheel_vx.mean(1), 0)*dt*blocked).sum())
    elapsed = float(t[end]-t[begin])
    advance = float(root[end, 0]-root[begin, 0])
    rear = np.asarray(arrays['wheel_position'])[:, 2:]
    rear_bottom = rear[..., 2]-radius
    rear_wait = moving[:, None] & (np.arange(len(t)) >= begin)[:, None] & (rear[..., 0] < start_x-.22)
    rear_early = bool((rear_wait & (rear_bottom > .035)).any())
    phase_times = support_times(t, np.asarray(arrays['supported_tread']))
    b2 = phase_times.get('2', [None]*4)[:2]
    next_lead = int(np.argmin(b2)) if all(v is not None for v in b2) else None
    coordinated = bool(lead_side == next_lead) if valid_support else False
    safe = bool(from_lower and ordered and coordinated and not rear_early and twin_seconds <= .02
                and pitch[scope].max() <= 12. and np.abs(rate[scope]).max() <= 90.
                and blocked_distance <= .05 and elapsed <= 3. and advance >= .05)
    return dict(protocol=PROTOCOL, scope='from_lower' if from_lower else 'unmeasured_or_starts_on_stair',
        alternating=ordered, safe_entry=safe, lead_side=lead_side if valid_support else None,
        lead_support_retained=retained_support,
        rear_lift_before_riser=rear_early, lead_matches_following=coordinated,
        support_times=phase_times,
        lift_times=[float(t[i]) if i is not None else None for i in first_lift],
        b1_support_onset_times=[float(t[i]) if i is not None else None for i in first_support],
        b1_confirmed_times=[float(t[i]) if i is not None else None for i in confirmed],
        twin_air_seconds=twin_seconds, blocked_body_advance_m=blocked_distance,
        nose_down_peak_deg=float(pitch[scope].max()),
        pitch_peak_abs_deg=float(np.abs(pitch[scope]).max()),
        pitch_rate_peak_deg_s=float(np.abs(rate[scope]).max()),
        entry_elapsed_seconds=elapsed, entry_body_progress_m=advance,
        approach_begin_time=float(t[begin]), entry_end_time=float(t[end]))
