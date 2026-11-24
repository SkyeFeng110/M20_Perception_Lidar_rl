#pragma once
#include "yaw_command_limits.hpp"
#include "joint_safety_limits.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace m20 {
// Owned by the single policy worker; reset only before the worker starts.
class YawCommandRamp {
    double value_=0,last_time_=0;
public:
    void reset(double now){value_=0;last_time_=now;}
    float sample(float request,double now){
        if(!std::isfinite(request)||!std::isfinite(now)||now<last_time_)
            return std::numeric_limits<float>::quiet_NaN();
        const double dt=std::min(now-last_time_,joint_limits::policy_period_s);
        last_time_=now;
        if(request==0 || (value_*request>=0 && std::abs(request)<std::abs(value_)))
            value_=request; // stopping and reductions are never delayed
        else if(value_*request<0)value_=0;
        else {
            const double step=yaw_limits::acceleration*dt;
            value_+=std::clamp(double(request)-value_,-step,step);
        }
        return float(value_);
    }
};
}
