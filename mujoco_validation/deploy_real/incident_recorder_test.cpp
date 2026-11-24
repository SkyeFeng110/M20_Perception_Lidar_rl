#include "incident_recorder.hpp"
#include "joint_command_guard.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){try{
    if(argc!=2 && argc!=3)throw std::runtime_error("usage: incident_recorder_test EMPTY_OWNED_DIR [io_failure|freeze]");
    const std::string dir=argv[1];
    if(!std::filesystem::is_empty(dir))throw std::runtime_error("test requires empty directory");
    if(argc==3 && std::string(argv[2])=="startup"){
        m20::incident::Recorder recorder(dir);m20::incident::StateFaultGate gate;
        m20::incident::Record r;r.kind=m20::incident::Kind::State;r.safe_mode=3;
        r.fault=gate.observe(r.safe_mode);
        if(r.fault || !recorder.put(r))throw std::runtime_error("startup wait classified as fault");
        std::this_thread::sleep_for(std::chrono::duration<double>(m20::joint_limits::incident_post_fault_s+0.1));
        if(!recorder.healthy() || recorder.frozen() || std::filesystem::exists(std::filesystem::path(dir)/"first_fault.json"))
            throw std::runtime_error("startup wait froze recorder");
        if(gate.observe(0))throw std::runtime_error("healthy transition classified as fault");
        r.fault=gate.observe(3);m20::incident::reason(r,"runtime_safety_stop");
        if(!r.fault || !recorder.put(r))throw std::runtime_error("runtime stop not recorded");
        std::this_thread::sleep_for(std::chrono::duration<double>(m20::joint_limits::incident_post_fault_s+0.1));
        if(!recorder.frozen() || recorder.healthy())throw std::runtime_error("runtime stop did not freeze recorder");
        m20::incident::StateFaultGate early;
        if(!early.observe(3,true))throw std::runtime_error("startup joint fault ignored");
        std::cout<<"INCIDENT_STARTUP=PASS startup_wait_healthy_runtime_stop_frozen_joint_fault_retained\n";return 0;
    }
    if(argc==3 && std::string(argv[2])=="freeze"){
        m20::incident::Recorder recorder(dir);m20::incident::Record r;
        r.fault=true;r.kind=m20::incident::Kind::Command;r.mono_s=m20::incident::mono();
        m20::incident::reason(r,"test_first_fault");recorder.put(r);
        std::this_thread::sleep_for(std::chrono::duration<double>(m20::joint_limits::incident_post_fault_s+0.05));
        if(!recorder.frozen() || recorder.healthy())throw std::runtime_error("first-fault history not frozen");
        if(recorder.put(r))throw std::runtime_error("frozen recording overwritten");
        std::ifstream f(std::filesystem::path(dir)/"first_fault.json");nlohmann::json first;f>>first;
        if(first["reason"]!="test_first_fault")throw std::runtime_error("first trigger lost");
        std::cout<<"INCIDENT_FREEZE=PASS first_trigger_and_history_preserved_restart_required\n";return 0;
    }
    if(argc==3){
        if(std::string(argv[2])!="io_failure")throw std::runtime_error("unknown failure mode");
        // Real write failure, not a mocked healthy flag. Never used in control.
        std::signal(SIGXFSZ,SIG_IGN);struct rlimit limit{65536,65536};
        if(setrlimit(RLIMIT_FSIZE,&limit)!=0)throw std::runtime_error("cannot set test write quota");
        m20::incident::Recorder recorder(dir);
        for(int i=0;i<2000 && recorder.healthy();++i){
            m20::incident::Record r;r.kind=m20::incident::Kind::Command;r.fault=true;
            recorder.put(r);std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if(recorder.healthy())throw std::runtime_error("write failure not detected");
        m20::joint_safety::Guard guard;auto c=m20::joint_safety::damping();c[0][0]=80;
        m20::joint_safety::Feedback f;f.receipt_mono_s=10;
        if(guard.validate(c,f,10,recorder.healthy()).reason!=m20::joint_safety::Reason::Recorder ||
           !guard.validate(m20::joint_safety::damping(),f,10,false).ok())
            throw std::runtime_error("write failure did not reject high gain and preserve damping");
        std::cout<<"INCIDENT_IO_FAILURE=PASS real_write_failure_active_rejected_damping_allowed\n";
        return 0;
    }
    int submitted=0;
    {
        m20::incident::Recorder recorder(dir);
        bool locked=false;try{m20::incident::Recorder another(dir);}catch(...){locked=true;}
        if(!locked)throw std::runtime_error("concurrent writer accepted");
        std::vector<std::thread> producers;
        for(int p=0;p<3;++p)producers.emplace_back([&,p]{
            for(int k=0;k<100;++k){m20::incident::Record r;r.kind=m20::incident::Kind::Command;
                r.inference_id=p*100+k;r.mono_s=m20::incident::mono();r.wall_s=m20::incident::wall();
                r.q[0]=float(p);r.sdk_command[1]=float(k);m20::incident::reason(r,"thread_fixture");
                r.fault=k==99;if(!recorder.put(r))std::abort();}
        });
        for(auto&t:producers)t.join();
        submitted=300;
        if(!recorder.healthy())throw std::runtime_error("healthy recording failed");
    }
    int records=0,ends=0;
    for(const auto&e:std::filesystem::directory_iterator(dir)){
        if(e.path().extension()!=".jsonl")continue;
        if(e.file_size()>m20::joint_limits::incident_segment_bytes)throw std::runtime_error("segment quota exceeded");
        std::ifstream stream(e.path());std::string line;
        while(std::getline(stream,line)){auto j=nlohmann::json::parse(line);
            if(j["event"]=="command_attempt")++records;
            if(j["event"]=="session_end"){++ends;if(!j["complete"].get<bool>())throw std::runtime_error("incomplete recorder");}}
    }
    // Small-ledger rotation tests retain a bounded suffix, not all records.
    if(records==0 || records>submitted || ends!=1)throw std::runtime_error("record/frame footer mismatch");
    std::cout<<"INCIDENT_RECORDER=PASS submitted="<<submitted<<" retained="<<records<<" no_motor_io\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
