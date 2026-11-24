// Static-angle diagnosis only. Uses the production guard; no ROS or motor I/O.
// Input: 16 recorded SDK joint angles in FL/FR/HL/HR hip/thigh/calf/wheel order.
// Velocity and effort are zero in this replay, so this is not dynamics proof.
#include "joint_command_guard.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
    if(argc!=2)return 2;
    std::ifstream input(argv[1]);
    m20::joint_safety::Feedback f;f.receipt_mono_s=10;
    for(auto& q:f.q)if(!(input>>q))return 2;
    auto c=m20::joint_safety::damping();
    for(int i=0;i<16;++i){c[i][1]=f.q[i];if(i%4!=3)c[i][0]=80;}
    m20::joint_safety::Guard raw;
    auto r=raw.validate(c,f,10,true);
    std::cout<<"RECORDED_POSE_RAW_TARGET="<<m20::joint_safety::name(r.reason)
             <<" joint="<<r.joint<<" value="<<r.value<<" limit="<<r.limit<<'\n';
    for(int i=0;i<16;++i)if(i%4!=3)
        c[i][1]=std::max(m20::joint_limits::lower(i),std::min(m20::joint_limits::upper(i),c[i][1]));
    m20::joint_safety::Guard clipped;
    r=clipped.validate(c,f,10,true);
    std::cout<<"RECORDED_POSE_CLIPPED_TARGET="<<m20::joint_safety::name(r.reason)
             <<" joint="<<r.joint<<" value="<<r.value<<'\n';
}
