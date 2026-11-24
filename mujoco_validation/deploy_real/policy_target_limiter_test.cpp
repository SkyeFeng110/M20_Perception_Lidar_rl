#include "policy_target_limiter.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace m20::joint_safety;
static int checks=0;
static void check(bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);}
static Command support(){auto c=damping();for(int i=0;i<16;++i)if(i%4!=3)c[i][0]=80;return c;}
int main(){try{
    const auto base=support();PolicyTargetLimiter l;check(l.seed(base,10),"standing seed");
    auto req=base;for(int i=0;i<16;++i)req[i][1]=i%2?100:-100;
    auto first=l.project(req,11);
    for(int i=0;i<16;++i)if(i%4!=3) {
        check(std::abs(first[i][1])<=m20::target_limits::max_rate_rad_s[i%4]*.005+1e-7,"stall accumulated allowance");
    }else check(first[i][1]==req[i][1],"wheel position changed");
    auto discarded=l.project(req,12);check(discarded==first,"unpublished candidate advanced target");
    l.committed(first,12);check(l.project(req,12.005)!=first,"successful publication did not advance");
    auto bad=base;bad[0][1]=NAN;check(std::isnan(l.project(bad,12.01)[0][1]),"nonfinite request hidden");
    check(l.project(damping(),NAN)==damping(),"damping delayed");
    PolicyTargetLimiter empty;Guard g;Feedback f;f.receipt_mono_s=10;
    check(g.validate(empty.project(base,10),f,10).reason==Reason::Nonfinite,"missing handoff failed open");
    check(!l.seed(bad,20),"invalid handoff accepted");
    check(l.seed(base,20),"valid reseed");
    check(l.project(req,19)[0][0]>0&&std::isnan(l.project(req,19)[0][1]),"clock rollback accepted");
    std::mt19937 rng(45200);std::uniform_real_distribution<float> value(-5,5);
    auto previous=base;double t=30;check(l.seed(previous,t),"random seed");
    for(int tick=0;tick<5000;++tick) {
        t+=.02;req=base;for(int i=0;i<16;++i)req[i][1]=value(rng);
        const auto out=l.project(req,t);
        for(int i=0;i<16;++i)if(i%4!=3) {
            check(out[i][1]>=m20::joint_limits::lower(i)+m20::target_limits::margin_rad[i%4]-1e-6&&
                  out[i][1]<=m20::joint_limits::upper(i)-m20::target_limits::margin_rad[i%4]+1e-6,"protected angle exceeded");
            check(std::abs(out[i][1]-previous[i][1])<=m20::target_limits::max_rate_rad_s[i%4]*.005+1e-6,"publication slew exceeded");
        }
        if(tick%3){l.committed(out,t);previous=out;}
    }
    // Shaping does not replace the independent tracking-error gate.
    auto c=base;c[0][0]=1000;c[0][1]=.06f;Guard hard;hard.committed(c,9.99);f.receipt_mono_s=10;
    check(hard.validate(c,f,10).reason==Reason::TrackingError,"tracking protection missing");
    std::cout<<"POLICY_TARGET_LIMITER=PASS checks="<<checks<<" scope=no_motor_io\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
