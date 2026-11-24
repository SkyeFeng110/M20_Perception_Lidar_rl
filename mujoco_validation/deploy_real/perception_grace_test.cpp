// Virtual-clock production store/commit tests. No ROS, DDS or motor API.
#include "v11_observation_store.hpp"
#include "perception_grace_limits.hpp"
#include <iostream>
using namespace m20::v11;
using nlohmann::json;
static unsigned checks=0;
static void check(bool ok,const char* message){++checks;if(!ok)throw std::runtime_error(message);}
static json measured(double stamp,unsigned seq,unsigned pub,float h=.16f){
    return {{"schema",1},{"source","gos_staircase"},{"contract","v11_signed_edges_v2"},
        {"layout","B1,signedH1,D1,B2,signedH2,D2"},{"frame","gos_odom"},
        {"epoch","occlusion-test"},{"source_seq",seq},{"publish_seq",pub},
        {"source_stamp_s",stamp},{"features",h==0?json::array({0.,0.,0.,0.,0.,0.}):json::array({1.,h,.5,0.,0.,0.})},
        {"source_valid",true},{"processing_ok",true},{"valid",true},{"policy_input",true},
        {"observation_status",h==0?"no_stairs_detected":"stairs_detected"},
        {"tracker_applied",true},{"fixture",false}};
}
static json bundle(const json& p){
    return {{"schema",2},{"current",p},{"retained",p["features"][0]==1?
        json{{"packet",p},{"expires_stamp_s",p["source_stamp_s"].get<double>()+.2}}:json(nullptr)}};
}
static json unavailable(json p,unsigned pub){
    p["publish_seq"]=pub;p["features"]=std::array<float,6>{};
    for(auto k:{"valid","policy_input","source_valid","processing_ok"})p[k]=false;
    p["observation_status"]="invalid";p["reason"]="source_timeout";
    return {{"schema",2},{"current",p},{"retained",nullptr}};
}
static bool put(ObservationStore& s,const json&p,double wall,double mono,bool unique=true){
    return s.UpdatePacket(p.dump(),"v11_signed_edges_v2",wall,mono,unique);
}
static CommitRequest request(const ObservationSnapshot& s){
    return {s.lease,s.steady_deadline_seconds,s.valid,s.lease.fault_generation};
}
int main(){try{
    const double duration=m20::perception_limits::loss_timeout_s;
    for(float h:{.16f,-.16f,0.f}){
        ObservationStore s(ObservationSource::Staircase,.3,duration);
        check(!s.Read(10).valid,"startup without a measurement must stop");
        const auto p=measured(1000,1,1,h);
        check(put(s,bundle(p),1000.01,10.01),"initial healthy source");
        auto original=s.Read(10.01);const auto initial_generation=original.lease.fault_generation;
        auto held=s.Read(10.31);
        check(held.valid&&held.selection==FeatureSelection::DegradedHold,"silent loss holds");
        check(held.features==original.features,"signed stair/flat feature preserved");
        check(std::abs(held.steady_deadline_seconds-(10+duration))<1e-8,"deadline includes initial source age");
        check(held.source.source_stamp_s==1000&&held.age_seconds>.3,"provenance not restamped");
        if(h!=0)check(s.Read(10.21).selection==FeatureSelection::DegradedHold,"no zero flicker at geometry expiry");
        check(ClassifyLease(original.lease,original.steady_deadline_seconds,true,10.31)==LeaseStatus::GeometryExpired,
              "in-flight normal lease must reselect rather than latch on a soft loss");
        unsigned published=0;
        check(s.GuardedCommit(request(held),[]{return 10.32;},[&](const auto&){++published;}).committed,
              "policy publication continues in short loss");
        for(unsigned i=2;i<12;++i){
            check(!put(s,unavailable(p,i),1000+.3+.01*i,10+.3+.01*i),"invalid source stays labelled invalid");
            auto v=s.Read(10.5);
            check(v.valid&&v.lease.fault_generation==initial_generation,"authentic unavailability is soft");
            check(v.steady_deadline_seconds==held.steady_deadline_seconds,"invalid tick cannot renew hold");
        }
        check(s.Read(held.steady_deadline_seconds-.001).valid&&!s.Read(held.steady_deadline_seconds).valid,"exact long-loss boundary");
        check(!s.GuardedCommit(request(held),[&]{return held.steady_deadline_seconds;},[&](const auto&){++published;}).committed&&published==1,
              "publisher lock delay cannot outlive grace");
        const auto p2=measured(1000+duration-.1,2,12,-h);
        check(put(s,bundle(p2),1000+duration-.09,10+duration-.09),"genuinely new frame recovers");
        auto fresh=s.Read(10+duration-.09);
        check(fresh.valid&&fresh.selection!=FeatureSelection::DegradedHold,"recovery uses fresh observation");
        check(fresh.features[1]==-h,"recovery updates signed geometry");
        check(s.Read(10+duration+.25).valid,"new source earns independent bounded grace");
        s.Reset();check(!s.Read(10+duration+.25).valid,"reset cannot reuse cached terrain");
    }
    {
        ObservationStore s(ObservationSource::Staircase,.3,duration);
        auto p=measured(1000,1,1);check(put(s,bundle(p),1000.01,10.01),"gap seed");
        const auto admitted=s.Read(10.01).lease.fault_generation;
        check(!put(s,unavailable(p,2),1000.31,10.31),"gap soft event");
        p=measured(1000+duration+.01,2,3);
        check(put(s,bundle(p),1000+duration+.02,10+duration+.02),"late new frame is measurable");
        const auto fresh=s.Read(10+duration+.02);
        check(fresh.lease.fault_generation>admitted,"recovery cannot erase an unsampled long loss");
        auto old_episode=request(fresh);old_episode.admitted_fault_generation=admitted;
        check(!s.GuardedCommit(old_episode,[&]{return 10+duration+.02;},[](const auto&){}).committed,
              "running episode stops after long loss even if recovery raced its next tick");
    }
    // Repeated healthy publication, backward wall steps and hard faults cannot
    // renew cached measurement age or revive an in-flight pre-fault command.
    for(const std::string fault:{"malformed","publisher","restart","fixture","contract","unknown_unavailability"}){
        ObservationStore s(ObservationSource::Staircase,.3,duration);
        auto p=measured(1000,1,1);check(put(s,bundle(p),1000.01,10.01),"hard test seed");
        auto original=s.Read(10.01);
        p["publish_seq"]=2u;check(put(s,bundle(p),1000.0,10.1),"repeated healthy source with wall step");
        check(std::abs(s.Read(10.31).steady_deadline_seconds-(10+duration))<1e-8,"repeated healthy source cannot extend deadline");
        auto bad=bundle(p);bad["current"]["publish_seq"]=3u;
        if(fault=="malformed")bad=json::object();
        if(fault=="restart"){bad["current"]["epoch"]="new";bad["retained"]["packet"]["epoch"]="new";}
        if(fault=="fixture")bad["current"]["fixture"]=true;
        if(fault=="contract")bad["current"]["contract"]="wrong";
        if(fault=="unknown_unavailability"){bad=unavailable(p,3);bad["current"]["reason"]="identity_failed";}
        check(!put(s,bad,1000.12,10.12,fault!="publisher"),"hard fault rejected");
        check(!s.Read(10.13).valid,"hard fault has no grace");
        check(!s.GuardedCommit(request(original),[]{return 10.13;},[](const auto&){}).committed,
              "hard fault cancels in-flight command");
    }
    std::cout<<"PERCEPTION_GRACE=PASS checks="<<checks<<" scope=virtual_clock_no_motor_io\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
