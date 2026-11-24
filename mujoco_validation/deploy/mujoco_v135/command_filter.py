"""Normal forward-command slew contract, shared by training and validation.

Emergency stops and policy exit bypass this normal motion-command filter.
The arithmetic accepts numpy arrays or torch tensors without importing torch.
"""
FORWARD_ACCELERATION = 1.5  # m/s^2; full 1 m/s remains available
FORWARD_DECELERATION = 10.0 / 3.0
POLICY_PERIOD = .02


def forward_slew(current, requested, dt, *, where, clip, acceleration=FORWARD_ACCELERATION):
    # Reverse through zero. Do not consume a delayed worker's entire elapsed
    # time in one update and thereby jump straight to full command.
    dt = min(max(float(dt), 0.), POLICY_PERIOD)
    braking = (current * requested < 0) | (abs(requested) < abs(current))
    goal = where(current * requested < 0, requested * 0., requested)
    rate = where(braking, FORWARD_DECELERATION, acceleration)
    return current + clip(goal-current, -rate*dt, rate*dt)
