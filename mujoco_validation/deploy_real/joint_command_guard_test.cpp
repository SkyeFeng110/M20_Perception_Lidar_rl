#include "joint_command_guard.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace m20::joint_safety;
static int checks=0;
void require(bool ok,const char*message){++checks;if(!ok)throw std::runtime_error(message);}
Feedback feedback(const Command&c,double now=10){Feedback f;f.receipt_mono_s=now;for(int i=0;i<16;++i)f.q[i]=c[i][1];return f;}
Command active(){auto c=damping();for(int i=0;i<16;++i)if(i%4!=3)c[i][0]=80;return c;}
int main(){try{
    const auto base=active();
    for(int i=0;i<16;++i){if(i%4==3)continue;
        for(int side:{-1,1}){
            auto c=base;const float edge=side<0?m20::joint_limits::lower(i):m20::joint_limits::upper(i);
            c[i][1]=edge;Guard good;require(good.validate(c,feedback(c),10).ok(),"inclusive target boundary");
            c[i][1]=std::nextafter(edge,side<0?-INFINITY:INFINITY);
            Guard bad;auto r=bad.validate(c,feedback(base),10);
            require(r.reason==Reason::Angle && r.joint==i,"every joint rejects finite out of range");
            require(bad.validate(base,feedback(base),10).reason==Reason::Latched,"valid action cannot clear fault");
            require(bad.validate(damping(),Feedback{},NAN,false).ok(),"damping cannot be blocked");
        }
        for(int column=0;column<5;++column)for(float value:{NAN,INFINITY,-INFINITY}){
            auto c=base;c[i][column]=value;Guard g;
            require(!g.validate(c,feedback(base),10).ok(),"nonfinite command rejected");
        }
    }
    // Continuous wheel position must not inherit a leg angle limit.
    for(int i=3;i<16;i+=4){auto c=base;c[i][1]=1.e6f;Guard g;require(g.validate(c,feedback(c),10).ok(),"continuous wheel");}
    for(double age:{0.05,1.,-0.01}){Guard g;auto f=feedback(base);f.receipt_mono_s-=age;require(g.validate(base,f,10).reason==Reason::FeedbackStale,"stale/future feedback");}
    for(int i=0;i<16;++i){Guard g;auto f=feedback(base);f.qd[i]=NAN;require(g.validate(base,f,10).reason==Reason::FeedbackNonfinite,"invalid feedback");}
    {Guard g;require(g.validate(base,feedback(base),10,false).reason==Reason::Recorder,"missing recorder rejects active");}
    {Guard g;auto c=damping();c[0][1]=NAN;require(g.validate(c,Feedback{},NAN,false).ok(),"unused passive position");}
    for(int i=0;i<16;++i){if(i%4==3)continue;
        auto c=base;Guard g;auto f=feedback(c);require(g.validate(c,f,10).ok(),"initial active");g.committed(c,10);
        c[i][1]=float(m20::joint_limits::target_rate_rad_s[i%4]*0.005*1.01);f.receipt_mono_s=10.005;
        require(g.validate(c,f,10.005).reason==Reason::TargetRate,"rate violation after actual publish");
        Guard startup;require(startup.validate(c,f,10.005).reason==Reason::TargetRate,"first command compared to feedback");
    }
    {Guard g;auto c=base;auto f=feedback(c);g.committed(c,10);f.receipt_mono_s=20;c[1][1]=1.1f;
        require(g.validate(c,f,20).reason==Reason::TargetRate,"long pause cannot grant arbitrary jump");}
    {Guard g;auto c=base;auto f=feedback(c);g.committed(c,10);c[1][1]=0.1f;f.receipt_mono_s=10.02;
        require(g.validate(c,f,10.02).ok(),"normal target increment");
        // No committed(): a geometry discard must not advance target history.
        c[1][1]=1.2f;require(g.validate(c,f,10.02).reason==Reason::TargetRate,"discard retains last published target");}
    {Guard g;auto c=base;auto f=feedback(base);c[0][0]=1000;c[0][1]=0.1f;g.committed(c,9.99);
        require(g.validate(c,f,10).reason==Reason::TrackingError,"legal angle with excessive position error");}
    {Guard g;auto c=base;auto f=feedback(c);c[1][3]=23;
        require(g.validate(c,f,10).reason==Reason::Velocity,"velocity command limit");}
    {Guard g;auto c=base;auto f=feedback(c);c[2][4]=77;
        require(g.validate(c,f,10).reason==Reason::Feedforward,"feedforward command limit");}
    {Guard g;auto c=base;auto f=feedback(c);c[1][0]=-1;
        require(g.validate(c,f,10).reason==Reason::NegativeGain,"negative gain");}
    if(m20::joint_limits::tracking_error_max_rad[12]>0){
        const bool all_hipx=m20::joint_limits::tracking_error_max_rad[0]>0;
        for(int i=0;i<16;++i)if(i%4!=3){
            const bool overridden=i==12 || (all_hipx && i%4==0);
            require(std::abs(m20::joint_limits::tracking_error_max_rad[i]-(overridden?.65:0.))<1e-12,"named angular threshold or override isolation");
            const double expected_proxy=(overridden?52.0:(i%4==1?95.0:(i%4==2?85.0:m20::joint_limits::leg_torque_nm[i%4])));
            require(std::abs(m20::joint_limits::tracking_error_proxy_nm[i]-expected_proxy)<1e-9,"named proxy ceiling or unchanged joint budget");
        }
        for(int joint:{0,4,8,12}){
            if(m20::joint_limits::tracking_error_max_rad[joint]>0){
                // Choose a sign inside the asymmetric named angle envelope.
                const float sign=(joint/4)%2?-1.f:1.f;
                for(float gain:{40.f,80.f,160.f})for(int side:{-1,1}){
                    auto c=base;auto f=feedback(base);c[joint][0]=gain;
                    const double allowed=std::min(.65,52.0/gain);
                    const double target=double(m20::joint_limits::lower(joint))+.2+allowed;
                    const double error=allowed+side*.0001;
                    c[joint][1]=float(target); f.q[joint]=float(target-error); Guard g;g.committed(c,9.99);
                    const auto result=g.validate(c,f,10);
                    require(side<0?result.ok():result.reason==Reason::TrackingError,"HipX angular boundary or gain ceiling");
                }
            }else{
                auto c=base;auto f=feedback(base);c[joint][1]=joint==4?-.414f:.414f;Guard g;g.committed(c,9.99);
                require(g.validate(c,f,10).reason==Reason::TrackingError,"unselected HipX lost original threshold");
            }
            auto c=base;auto f=feedback(base);c[joint][4]=32.5f;Guard g;
            require(g.validate(c,f,10).reason==Reason::Feedforward,"HipX FF torque budget widened");
        }
        for(int joint=0;joint<16;++joint)if(joint%4==1||joint%4==2){
            auto c=base;auto f=feedback(base);c[joint][1]=1.2f;Guard g;g.committed(c,9.99);
            require(g.validate(c,f,10).reason==Reason::TrackingError,"HipY or knee tracking budget changed");
        }
        if(all_hipx)std::cout<<"ALL_HIPX_TRACKING_OVERRIDE=PASS joints=0,4,8,12 max_error_rad=.65 max_proxy_nm=52.0 ff_limit_nm=32.4\n";
        else std::cout<<"HR_HIPX_TRACKING_OVERRIDE=PASS max_error_rad=.65 max_proxy_nm=52.0 ff_limit_nm=32.4\n";
    }
    for(int joint:{2,6,10,14}) {
        require(std::abs(m20::joint_limits::tracking_error_proxy_nm[joint]-85.)<1e-9,"four knee proxy override missing");
        require(m20::joint_limits::tracking_error_max_rad[joint]==0,"knee angular guard unexpectedly changed");
        require(m20::joint_limits::leg_torque_nm[2]==76.4,"knee feedforward budget changed with proxy");
        for(float gain:{40.f,80.f,160.f})for(float sign:{-1.f,1.f})for(int side:{-1,1}) {
            auto c=base;auto f=feedback(base);c[joint][0]=gain;
            c[joint][1]=sign*float(85./gain+side*.0001);Guard g;g.committed(c,9.99);
            const auto result=g.validate(c,f,10);
            require(side<0?result.ok():result.reason==Reason::TrackingError,"knee proxy boundary across gains/directions");
        }
        auto c=base;auto f=feedback(base);c[joint][4]=76.5f;Guard g;
        require(g.validate(c,f,10).reason==Reason::Feedforward,"knee FF budget widened");
    }
    std::cout<<"ALL_KNEE_PROXY=PASS joints=2,6,10,14 proxy_nm=85 kp80_error_rad=1.0625 ff_limit_nm=76.4\n";
    std::cout<<"JOINT_COMMAND_GUARD=PASS checks="<<checks<<" no_motor_io\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<'\n';return 1;}}
