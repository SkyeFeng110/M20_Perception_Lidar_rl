#include "yaw_command_ramp.hpp"
#include <iostream>
#include <stdexcept>
static unsigned checks=0;
static void check(bool ok,const char*s){++checks;if(!ok)throw std::runtime_error(s);}
int main(){try{
    const double period=m20::joint_limits::policy_period_s;
    const double step=m20::yaw_limits::acceleration*period;
    for(float request:{-.6f,.6f}){
        m20::YawCommandRamp ramp;ramp.reset(10);
        check(ramp.sample(request,10+period)==request,"full stick onset was delayed");
        ramp.reset(10);
        float previous=0;
        for(int tick=1;tick<=60;++tick){
            const float actual=ramp.sample(request,10+tick*period);
            check(std::abs(actual-previous)<=step+1e-6,"turn onset exceeded slew limit");
            check(actual*request>=0&&std::abs(actual)<=std::abs(request),"direction or magnitude changed");previous=actual;
        }
        check(std::abs(previous-request)<1e-6,"full stick magnitude was reduced");
        check(ramp.sample(request/2,11.21)==request/2,"reduction delayed");
        check(ramp.sample(-request,11.22)==0,"reversal did not pass through zero");
        check(std::abs(ramp.sample(-request,100))<=step+1e-6,"stall accumulated ramp allowance");
        check(ramp.sample(0,100.001)==0,"neutral stop delayed");
        ramp.reset(200);check(std::abs(ramp.sample(request,200.005))<=m20::yaw_limits::acceleration*.005+1e-6,"mode entry jump");
        check(!std::isfinite(ramp.sample(NAN,200.01)),"invalid input hidden");
        check(!std::isfinite(ramp.sample(request,199)),"clock rollback hidden");
    }
    std::cout<<"YAW_COMMAND_RAMP=PASS checks="<<checks<<" scope=no_motor_io\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
