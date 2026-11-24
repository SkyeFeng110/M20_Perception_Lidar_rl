#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace m20 {
// Same normal-forward-command contract as mujoco_v135/command_filter.py.
// Policy worker owns this state; emergency damping never goes through it.
class ForwardCommandRamp {
    float value_ = 0.f;
public:
    void reset() { value_ = 0.f; }
    float sample(float requested, double dt = .02) {
        if (!std::isfinite(requested) || !std::isfinite(dt) || dt < 0)
            return std::numeric_limits<float>::quiet_NaN();
        dt = std::min(dt, .02);
        const bool reverse = value_ * requested < 0;
        const bool braking = reverse || std::abs(requested) < std::abs(value_);
        const float goal = reverse ? 0.f : requested;
        const float step = float((braking ? 10. / 3. : 1.5) * dt);
        value_ += std::clamp(goal-value_, -step, step);
        return value_;
    }
};
}
