#pragma once
// No ROS dependency: this exact class is used by the final DDS publisher and
// by failure-injection tests. Caller serializes validate/publish/commit.
#include "joint_safety_limits.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace m20 { namespace joint_safety {
using Command = std::array<std::array<float,5>,16>; // kp,q,kd,qd,tau; SDK coordinates
struct Feedback {
    std::array<float,16> q{},qd{},tau{};
    double receipt_mono_s=0;
};
enum class Reason { Ok, Latched, Nonfinite, NegativeGain, Calibration, Angle,
    FeedbackStale, FeedbackNonfinite, FeedbackAngle, TargetRate, TrackingError,
    Velocity, Feedforward, Recorder, Shape, Publisher };
inline const char* name(Reason r) {
    switch(r) {
    case Reason::Ok:return "ok";case Reason::Latched:return "joint_fault_latched";
    case Reason::Nonfinite:return "nonfinite_command";case Reason::NegativeGain:return "negative_gain";
    case Reason::Calibration:return "invalid_calibration";case Reason::Angle:return "target_angle_limit";
    case Reason::FeedbackStale:return "feedback_stale";case Reason::FeedbackNonfinite:return "feedback_nonfinite";
    case Reason::FeedbackAngle:return "feedback_angle_limit";case Reason::TargetRate:return "target_rate_limit";
    case Reason::TrackingError:return "target_feedback_error";case Reason::Velocity:return "target_velocity_limit";
    case Reason::Feedforward:return "feedforward_torque_limit";case Reason::Recorder:return "incident_recorder_failed";
    case Reason::Shape:return "command_shape";
    case Reason::Publisher:return "publisher_exception";
    }return "unknown";
}
struct Result {Reason reason=Reason::Ok;int joint=-1;double value=0,limit=0;bool ok()const{return reason==Reason::Ok;}};
inline bool passive(const Command& c) {
    for(const auto& j:c)
        if(j[0]!=0 || !std::isfinite(j[2]) || j[2]<0 || j[3]!=0 || j[4]!=0)return false;
    return true; // position is unused when kp=0; canonicalized before publish
}
inline Command damping() {
    Command c{};
    for(int i=0;i<16;++i)c[i][2]=i%4==3?1.f:2.f;
    return c;
}
class Guard {
    bool latched_=false;
    Result first_{};
    std::array<double,16> last_q_{},last_time_{};
    std::array<bool,16> active_{};
public:
    bool latched()const{return latched_;}
    Result first_fault()const{return first_;}
    // Recovery is only invoked by the state machine after an explicit R2
    // request has already forced the controller into damping.  It clears the
    // software admission latch and target history; it does not publish a
    // command by itself.
    void reset_for_recovery() {
        latched_=false;
        first_=Result{};
        last_q_.fill(0.0);
        last_time_.fill(0.0);
        active_.fill(false);
    }
    Result fail(Reason reason,int joint=-1,double value=0,double limit=0) {
        Result r{reason,joint,value,limit};
        if(!latched_){first_=r;latched_=true;}
        return r;
    }
    // Validation never clears a latch. Only explicit FSM recovery can reset it.
    Result validate(const Command& c,const Feedback& f,double now,bool recorder_ok=true) {
        if(passive(c))return {};
        if(latched_)return {Reason::Latched,first_.joint};
        if(!recorder_ok)return fail(Reason::Recorder);
        for(int i=0;i<16;++i){
            for(float v:c[i])if(!std::isfinite(v))return fail(Reason::Nonfinite,i);
            if(c[i][0]<0 || c[i][2]<0)return fail(Reason::NegativeGain,i);
            // Wheels are continuous rotation; no position boundary is applied.
            if(i%4==3)continue;
            const auto& j=c[i];
            // Check every leg target, including a disabled leg in a mixed command.
            // Pure damping/disable already bypassed above.
            if(j[1]<joint_limits::lower(i) || j[1]>joint_limits::upper(i))
                return fail(Reason::Angle,i,j[1],j[1]<joint_limits::lower(i)?joint_limits::lower(i):joint_limits::upper(i));
            if(std::abs(j[3])>joint_limits::leg_velocity_rad_s[i%4])return fail(Reason::Velocity,i,j[3],joint_limits::leg_velocity_rad_s[i%4]);
            if(std::abs(j[4])>joint_limits::leg_torque_nm[i%4])return fail(Reason::Feedforward,i,j[4],joint_limits::leg_torque_nm[i%4]);
        }
        if(!std::isfinite(now) || !std::isfinite(f.receipt_mono_s) || f.receipt_mono_s<=0 ||
           now<f.receipt_mono_s || now-f.receipt_mono_s>=joint_limits::feedback_timeout_s)
            return fail(Reason::FeedbackStale);
        for(int i=0;i<16;++i){
            if(!std::isfinite(f.q[i]) || !std::isfinite(f.qd[i]) || !std::isfinite(f.tau[i]))
                return fail(Reason::FeedbackNonfinite,i);
            if(i%4==3)continue;
            if(f.q[i]<joint_limits::lower(i) || f.q[i]>joint_limits::upper(i))return fail(Reason::FeedbackAngle,i,f.q[i]);
            if(c[i][0]==0 && c[i][3]==0 && c[i][4]==0)continue;
            const double elapsed=active_[i]?now-last_time_[i]:joint_limits::control_tick_s;
            if(!std::isfinite(elapsed) || elapsed<=0)return fail(Reason::TargetRate,i);
            const double allowance=joint_limits::target_rate_rad_s[i%4]*std::min(elapsed,joint_limits::policy_period_s);
            const double delta=std::abs(double(c[i][1])-(active_[i]?last_q_[i]:f.q[i]));
            if(delta>allowance)return fail(Reason::TargetRate,i,delta,allowance);
            const double error_torque=std::abs(double(c[i][1])-f.q[i])*c[i][0];
            double error_limit=joint_limits::tracking_error_proxy_nm[i];
            // A named angular override never grants extra allowance if a
            // higher position gain is unexpectedly selected. Hardware/FF
            // torque budgets remain the independent leg_torque_nm constants.
            if(joint_limits::tracking_error_max_rad[i]>0)
                error_limit=std::min(error_limit,joint_limits::tracking_error_max_rad[i]*c[i][0]);
            if(error_torque>error_limit)return fail(Reason::TrackingError,i,error_torque,error_limit);
        }
        return {};
    }
    // Call only after the actual publish succeeded, never for a rejected source
    // lease or target. This is deliberately separate from validate().
    void committed(const Command& c,double now) {
        if(passive(c)){active_.fill(false);return;}
        for(int i=0;i<16;++i){active_[i]=c[i][0]!=0 || c[i][3]!=0 || c[i][4]!=0;
            last_q_[i]=c[i][1];last_time_[i]=now;}
    }
};
}}
