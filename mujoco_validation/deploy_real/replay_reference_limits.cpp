// Recorded-feedback counterfactual only; never connects to ROS or a robot.
#include "joint_command_guard.hpp"
#ifdef REPLAY_SHAPED
#include "policy_target_limiter.hpp"
#endif
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
using nlohmann::json;
using namespace m20::joint_safety;
static Command command(const json& r){Command c{};for(int i=0;i<16;++i)for(int k=0;k<5;++k)c[i][k]=r["sdk_16x5"][i*5+k];return c;}
int main(int argc,char**argv){try{
    if(argc!=2)return 2;
    const std::string dir=argv[1];std::ifstream fault_file(dir+"/first_fault.json");json fault;fault_file>>fault;
    std::vector<json> rows;
    for(int i=0;i<4;++i){std::ifstream f(dir+"/segment_"+std::to_string(i)+".jsonl");std::string line;
        while(std::getline(f,line)){auto r=json::parse(line);if(r.value("event","")=="command_attempt"&&r["mono_s"]<=fault["mono_s"])rows.push_back(std::move(r));}}
    std::sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){return a["record_seq"].template get<unsigned long>()<b["record_seq"].template get<unsigned long>();});
    Guard guard;Command pending{},last{};bool have_pending=false,have_last=false,seeded=false;
    double last_time=0;unsigned checked=0,shaped=0;
    json failure=nullptr;double last_hipx=0;
#ifdef REPLAY_SHAPED
    PolicyTargetLimiter limiter;
#endif
    for(const auto& r:rows){
        const std::string reason=r["reason"];const double t=r["mono_s"];
        if(reason=="published"&&have_pending){
            guard.committed(pending,t);last=pending;last_time=t;have_last=true;have_pending=false;
#ifdef REPLAY_SHAPED
            if(r["inference_id"].get<unsigned>()>0)limiter.committed(pending,t);
#endif
            continue;
        }
        if(reason!="ok"&&reason!="target_feedback_error")continue;
        auto c=command(r);
        Feedback f;f.receipt_mono_s=r["feedback_receipt_mono_s"];
        f.q=r["feedback_q"].get<std::array<float,16>>();f.qd=r["feedback_qd"].get<std::array<float,16>>();f.tau=r["feedback_tau"].get<std::array<float,16>>();
#ifdef REPLAY_SHAPED
        if(r["inference_id"].get<unsigned>()>0&&!passive(c)){
            if(!seeded){if(!have_last||!limiter.seed(last,last_time))throw std::runtime_error("missing issued standing handoff");seeded=true;}
            const auto projected=limiter.project(c,t);if(c!=projected)++shaped;c=projected;
        }
#endif
        const auto result=guard.validate(c,f,t);++checked;last_hipx=c[12][1];
        if(!result.ok()){failure={{"reason",name(result.reason)},{"joint",result.joint},{"value",result.value},{"limit",result.limit},
            {"record_seq",r["record_seq"]},{"inference_id",r["inference_id"]},{"mono_s",t}};break;}
        pending=c;have_pending=true;
    }
    std::cout<<json{{"checked_attempts",checked},{"shaped_attempts",shaped},{"first_failure",failure},
        {"last_hr_hipx_target",last_hipx},{"ledger_sha256",m20::joint_limits::ledger_sha256},
        {"scope","recorded feedback only through original first fault; changed-target dynamics and subsequent locomotion not simulated"}}.dump(2)<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
