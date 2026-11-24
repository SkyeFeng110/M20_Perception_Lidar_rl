#pragma once
#include "joint_command_guard.hpp"
#include "policy_target_limits.hpp"
#include <limits>

namespace m20::joint_safety {
// Policy worker only. Seed from the issued standing command, and advance only
// after a successful publication. Feedback never rebases a support target.
class PolicyTargetLimiter {
    Command last_{};
    double published_s_=0;
    bool seeded_=false;
    static Command invalid() {
        auto c=damping();c[0][0]=1;c[0][1]=std::numeric_limits<float>::quiet_NaN();return c;
    }
    static bool finite(const Command& c) {
        for(const auto& j:c)for(float v:j)if(!std::isfinite(v))return false;
        return true;
    }
public:
    template<class Matrix> static Command from_matrix(const Matrix& m) {
        if(m.rows()!=16||m.cols()!=5)return invalid();
        Command c{};for(int i=0;i<16;++i)for(int k=0;k<5;++k)c[i][k]=m(i,k);return c;
    }
    template<class Matrix> static void to_matrix(const Command& c,Matrix& m) {
        for(int i=0;i<16;++i)for(int k=0;k<5;++k)m(i,k)=c[i][k];
    }
    bool seed(const Command& issued,double now) {
        seeded_=false;
        if(!finite(issued)||!std::isfinite(now))return false;
        for(int i=0;i<16;++i)if(i%4!=3) {
            const auto a=i%4;
            const float lo=joint_limits::lower(i)+target_limits::margin_rad[a];
            const float hi=joint_limits::upper(i)-target_limits::margin_rad[a];
            if(issued[i][0]<=0||issued[i][2]<0||
               issued[i][1]<lo||issued[i][1]>hi)return false;
        }
        last_=issued;published_s_=now;seeded_=true;return true;
    }
    Command project(const Command& request,double now) const {
        // Invalid values/gains keep reaching the independent hard guard.
        if(passive(request)||!finite(request))return request;
        if(!seeded_||!std::isfinite(now)||now<=published_s_)return invalid();
        auto out=request;
        const double dt=std::min(now-published_s_,joint_limits::control_tick_s);
        for(int i=0;i<16;++i)if(i%4!=3) {
            const auto a=i%4;
            const float lo=joint_limits::lower(i)+target_limits::margin_rad[a];
            const float hi=joint_limits::upper(i)-target_limits::margin_rad[a];
            const double step=target_limits::max_rate_rad_s[a]*dt;
            const double value=std::clamp(request[i][1],lo,hi);
            out[i][1]=float(std::clamp(value,std::max(double(lo),double(last_[i][1])-step),
                                           std::min(double(hi),double(last_[i][1])+step)));
        }
        return out;
    }
    void committed(const Command& issued,double now) {
        if(passive(issued)){seeded_=false;return;}
        // An unexpected receipt cannot seed the next active target.
        seed(issued,now);
    }
};
}
