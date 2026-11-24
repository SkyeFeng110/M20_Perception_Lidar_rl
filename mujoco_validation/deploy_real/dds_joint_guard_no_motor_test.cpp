// Compile-time replacement of DDS command emission: real hardware interface,
// mapping and final guard, but no command publisher is even constructed.
#ifndef M20_DDS_NO_MOTOR_TEST
#error This fixture must never compile with real command emission
#endif
#include "state_base.h"
#include "hardware/m20_interface.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
class Capture : public M20Interface {
public:
    std::vector<drdds::msg::JointsDataCmd> sent;
    Capture():M20Interface("M20"){
        test_joint_sink_=[this](const auto&msg){sent.push_back(msg);};
    }
    void feedback(const MatXf&c,bool stale=false){
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        for(int i=0;i<16;++i){joint_pos_(i)=c(i,1);joint_vel_(i)=0;joint_tau_(i)=0;}
        joint_receipt_mono_s_=m20::incident::mono()-(stale?1.:0.);
    }
    void wire_fixture(){
        auto message=std::make_shared<std_msgs::msg::String>();
        message->data=nlohmann::json({{"schema",2},{"current",{{"source_seq",77},{"publish_seq",99},
            {"source_stamp_s",12345.25},{"epoch","wire-epoch"},{"reason","source_timeout"}}}}).dump();
        CacheIncidentWire(message,m20::incident::mono(),123);
    }
    void bad_calibration(){joint_config_[0].dir=NAN;}
    m20::joint_safety::Reason first_reason(){return joint_guard_.first_fault().reason;}
    void reset_source(){ResetV11Observation();}
    void hold_publish_lock(std::atomic<bool>&locked){
        std::lock_guard<std::mutex> lock(joint_publish_mutex_);locked=true;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    m20::v11::CommitRequest source(){
        const auto now=m20::incident::mono(),wall=m20::incident::wall();
        nlohmann::json p={{"schema",1},{"source","gos_staircase"},{"contract","v11_signed_edges_v2"},
            {"layout","B1,signedH1,D1,B2,signedH2,D2"},{"frame","gos_odom"},{"epoch","guard_fixture"},
            {"source_seq",1},{"publish_seq",1},{"source_stamp_s",wall},{"features",{1.,.16,.5,1.,.15,.8}},
            {"source_valid",true},{"valid",true},{"policy_input",true},{"tracker_applied",true},{"fixture",false}};
        if(!v11_store_.UpdatePacket(p.dump(),"v11_signed_edges_v2",wall,now,true))throw std::runtime_error("test source rejected");
        const auto s=v11_store_.Read(now);
        return {s.lease,s.steady_deadline_seconds,s.valid,s.lease.fault_generation};
    }
};
static int checks=0;
void require(bool ok,const char*s){++checks;if(!ok)throw std::runtime_error(s);}
MatXf active(){auto m=MatXf::Zero(16,5).eval();for(int i=0;i<16;++i){m(i,0)=i%4==3?0:80;m(i,2)=2;}return m;}
bool passive(const drdds::msg::JointsDataCmd&m){for(int i=0;i<16;++i){const auto&j=m.data.joints_data[i];
    if(j.kp!=0 || j.velocity!=0 || j.torque!=0 || !std::isfinite(j.position) || !std::isfinite(j.kd))return false;}return true;}
int main(int argc,char**argv){try{
    if(argc!=2 && argc!=3 && argc!=4)throw std::runtime_error("usage: dds_joint_guard_no_motor_test EMPTY_OWNED_INCIDENT_DIR [--joint INDEX]");
    setenv("M20_INCIDENT_DIR",argv[1],1);
    const auto alignment_path=(std::filesystem::path(argv[1])/"alignment.jsonl").string();
    setenv("M20_ALIGNMENT_TRACE_PATH",alignment_path.c_str(),1);
    setenv("M20_ALIGNMENT_SCOPE","offline_dds_guard_component_no_motor",1);
    setenv("M20_V11_FEATURE_CONTRACT","v11_signed_edges_v2",1);
    rclcpp::init(argc,argv);
    if(argc==3){
        require(std::string(argv[2])=="--perception","unknown test selector");
        Capture hw;hw.wire_fixture();types::UserCommand uc;uc.safe_control_mode=0;
        auto& recorder=m20::incident::Recorder::global();
        for(const char* trigger:{"fsm_perception_fault_damping","rl_run_source_fault_damping","source_or_inference_fault_damping","publisher_source_fault_damping"}){
            hw.RecordPerceptionFault(4,uc,trigger);
            std::this_thread::sleep_for(std::chrono::duration<double>(m20::joint_limits::incident_post_fault_s+.1));
            require(recorder.frozen()&&!recorder.healthy(),"perception stop did not freeze recorder");
            nlohmann::json first;std::ifstream stream(std::filesystem::path(argv[1])/"first_fault.json");stream>>first;
            require(first["event"]=="perception_fault"&&first["reason"]==trigger&&first["fault_trigger"]==true,
                    "perception first trigger lost or replaced by SDK event");
            require(first["safe_mode"]==0&&first.contains("fault_generation")&&first.contains("source_reason"),
                    "perception metadata missing");
            require(first["latest_wire_captured"]==true&&first["latest_wire"]["source_seq"]==77&&
                    first["latest_wire"]["reason"]=="source_timeout"&&first["latest_wire"]["source_stamp_s"]==12345.25,
                    "triggering source provenance missing");
            auto c=active();hw.feedback(c);hw.SetJointCommand(c);
            require(passive(hw.sent.back()),"frozen evidence allowed active output");
            require(hw.ResetJointCommandFaultForRecovery(),"R2 failed to archive perception incident and rearm");
            require(recorder.healthy()&&!std::filesystem::exists(std::filesystem::path(argv[1])/"first_fault.json"),
                    "R2 left old first trigger active");
        }
        int archives=0;for(const auto& e:std::filesystem::directory_iterator(argv[1]))
            if(e.is_directory()&&e.path().filename().string().rfind("incident_",0)==0)++archives;
        require(archives==4,"R2 did not preserve each perception incident");
        rclcpp::shutdown();std::cout<<"DDS_PERCEPTION_FREEZE=PASS triggers=4 archives=4 command_publisher_compiled_out\n";return 0;
    }
    if(argc==4){
    if(std::string(argv[2])!="--joint")throw std::runtime_error("unknown test selector");
    const int selected=std::stoi(argv[3]);
    if(selected<0 || selected>=16 || selected%4==3)throw std::runtime_error("invalid leg joint");
    for(int i=0;i<16;++i){if(i!=selected)continue;for(int side:{-1,1}){
        Capture hw;auto c=active();hw.feedback(c);
        c(i,1)=side<0?m20::joint_limits::lower(i)-0.001f:m20::joint_limits::upper(i)+0.001f;
        hw.SetJointCommand(c);
        require(hw.sent.size()==1 && passive(hw.sent.back()),"generic state unsafe target escaped final publisher");
        require(hw.HasJointCommandFault(),"joint fault not visible to FSM");
        require(hw.first_reason()==m20::joint_safety::Reason::Angle,"wrong final-publisher rejection cause");
        c=active();hw.feedback(c);hw.reset_source();hw.SetJointCommand(c);
        require(passive(hw.sent.back()),"source recovery cleared joint fault");
    }}
    rclcpp::shutdown();std::cout<<"DDS_JOINT_BOUND=PASS joint="<<selected<<" checks="<<checks<<" command_publisher_compiled_out\n";return 0;
    }
    {Capture hw;types::UserCommand uc;uc.safe_control_mode=3;hw.RecordControllerState(0,uc);
        std::this_thread::sleep_for(std::chrono::duration<double>(m20::joint_limits::incident_post_fault_s+0.1));
        require(m20::incident::Recorder::global().healthy(),"real startup wait froze recorder before standing");
        require(!std::filesystem::exists(std::filesystem::path(argv[1])/"first_fault.json"),"startup generated incident marker");
        uc.safe_control_mode=0;hw.RecordControllerState(0,uc);
        auto c=active();hw.feedback(c);hw.SetJointCommand(c);
        require(!hw.HasJointCommandFault() && !passive(hw.sent.back()),"startup wait blocked generic stand command");
        std::cout<<"DDS_STARTUP=PASS startup_wait_then_generic_stand command_publisher_compiled_out\n";}
    {Capture hw;auto c=active();hw.feedback(c);auto lease=hw.source();auto result=hw.SetPolicyJointCommand(c,lease);
        require(result.committed && !hw.HasJointCommandFault(),"valid policy command rejected");
        require(hw.sent.back().data.joints_data[0].kp==80,"valid target changed");}
    {Capture hw;auto c=active();hw.feedback(c);auto lease=hw.source();
        lease.content_deadline_s=m20::incident::mono()+0.01;
        std::atomic<bool> locked{false};std::thread holder([&]{hw.hold_publish_lock(locked);});
        while(!locked.load())std::this_thread::yield();
        auto result=hw.SetPolicyJointCommand(c,lease);holder.join();
        require(!result.committed && result.status==m20::v11::LeaseStatus::Fault && hw.sent.empty() && !hw.HasJointCommandFault(),
                "source expired behind publish lock but was submitted or joint-latched");}
    {Capture hw;auto c=active();hw.feedback(c);auto lease=hw.source();c(6,1)=3;
        auto result=hw.SetPolicyJointCommand(c,lease);
        require(!result.committed && result.status==m20::v11::LeaseStatus::Fault,"rejected target reported as committed");
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"policy target bypassed final guard");}
    {Capture hw;auto c=active();hw.feedback(c);auto lease=hw.source();lease.content_deadline_s=m20::incident::mono()-1;
        auto result=hw.SetPolicyJointCommand(c,lease);
        require(!result.committed && hw.sent.empty() && !hw.HasJointCommandFault(),"expired source advanced publisher");}
    {Capture hw;auto c=active();hw.feedback(c,true);hw.SetJointCommand(c);
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"stale feedback escaped");}
    {Capture hw;auto c=active();hw.feedback(c);hw.bad_calibration();hw.SetJointCommand(c);
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"invalid calibration escaped");
        c.setZero();c.col(2).setConstant(2);hw.SetJointCommand(c);
        require(passive(hw.sent.back()),"invalid calibration prevented damping");}
    {Capture hw;auto c=active();hw.feedback(c);c(0,1)=NAN;hw.SetJointCommand(c);
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"nonfinite target escaped");}
    {Capture hw;MatXf bad=MatXf::Zero(12,5);hw.SetJointCommand(bad);
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"bad shape escaped");}
    {Capture hw;auto c=active();hw.feedback(c);hw.SetJointCommand(c);
        c(1,1)=0.6f;hw.SetJointCommand(c);
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"target jump escaped");}
    for(int recovery=0;recovery<2;++recovery){Capture hw;auto c=active();hw.feedback(c);c(6,1)=3;hw.SetJointCommand(c);
        require(hw.HasJointCommandFault() && passive(hw.sent.back()),"fault setup failed");
        std::this_thread::sleep_for(std::chrono::duration<double>(m20::joint_limits::incident_post_fault_s+.1));
        require(m20::incident::Recorder::global().frozen(),"recorder did not freeze after fault");
        const auto before=hw.sent.size();
        require(hw.ResetJointCommandFaultForRecovery(),"explicit passive R2 reset failed");
        require(hw.sent.size()==before+1 && passive(hw.sent.back()),"R2 reset did not unload joints");
        for(const auto&j:hw.sent.back().data.joints_data)
            require(j.kp==0 && j.kd==0 && j.position==0 && j.velocity==0 && j.torque==0,"R2 retained joint damping");
        require(m20::incident::Recorder::global().healthy(),"R2 did not reopen frozen recorder");
        auto good=active();hw.feedback(good);hw.SetJointCommand(good);
        require(!hw.HasJointCommandFault() && !passive(hw.sent.back()),"R2 reset did not rearm guarded output");
        unsigned archives=0;
        for(const auto&e:std::filesystem::directory_iterator(argv[1]))
            if(e.is_directory() && e.path().filename().string().find("incident_")==0){
                ++archives;require(std::filesystem::exists(e.path()/"first_fault.json"),"recovery discarded trigger evidence");
            }
        require(archives==unsigned(recovery+1),"repeated recovery did not preserve each incident");}
    rclcpp::shutdown();
    std::cout<<"DDS_JOINT_GUARD=PASS checks="<<checks<<" command_publisher_compiled_out\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<'\n';return 1;}}
