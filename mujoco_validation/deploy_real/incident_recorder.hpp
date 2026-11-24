#pragma once
// Always-on flight recorder, independent of optional point-cloud/alignment
// capture. Producers only copy fixed records into a bounded MPSC queue.
#include "joint_safety_limits.hpp"
#include "json.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>

namespace m20 { namespace incident {
inline double mono(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline double wall(){return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();}
enum class Kind { Command, Feedback, Inference, State, PerceptionFault };
struct Record {
    Kind kind=Kind::Command;
    double mono_s=0,wall_s=0,feedback_mono_s=0,source_age_s=0,deadline_s=0;
    std::uint64_t inference_id=0,source_seq=0,publish_seq=0,damping_reset_seq=0;
    double remote_command_stamp_s=0;
    int joint=-1,state=-1,target_mode=-1,safe_mode=-1;
    bool accepted=false,latched=false,observation_valid=false,inferred=false;
    double violation_value=0,violation_limit=0;
    char reason[96]{},epoch[96]{},source_reason[96]{};
    std::uint64_t fault_generation=0,store_revision=0;
    double grace_deadline_s=0,wire_stamp_s=0,wire_receipt_mono_s=0;
    std::uint64_t wire_source_seq=0,wire_publish_seq=0,wire_store_revision=0;
    char wire_reason[96]{},wire_epoch[96]{};bool wire_captured=false;
    std::array<float,80> sdk_command{},hardware_command{};
    std::array<float,16> q{},qd{},tau{},raw_action{},processed_action{},dir{},offset{};
    std::array<float,63> obs{};
    std::array<float,9> imu{};
    std::array<float,3> joystick{};
    bool fault=false;
};
inline void reason(Record&r,const char*s){std::strncpy(r.reason,s,sizeof(r.reason)-1);}
inline nlohmann::json encode(const Record&r) {
    using J=nlohmann::json;
    J j={{"mono_s",r.mono_s},{"wall_s",r.wall_s},{"inference_id",r.inference_id},
        {"reason",r.reason},{"accepted",r.accepted},{"joint_fault_latched",r.latched},
        {"fault_trigger",r.fault}};
    if(r.kind==Kind::Command){j["event"]="command_attempt";j["sdk_16x5"]=r.sdk_command;
        j["hardware_16x5"]=r.hardware_command;j["feedback_q"]=r.q;j["feedback_qd"]=r.qd;j["feedback_tau"]=r.tau;
        j["feedback_receipt_mono_s"]=r.feedback_mono_s;j["hardware_dir"]=r.dir;j["hardware_offset"]=r.offset;
        j["violation_joint"]=r.joint;j["violation_value"]=r.violation_value;j["violation_limit"]=r.violation_limit;
    }else if(r.kind==Kind::Feedback){j["event"]="joint_feedback";j["q"]=r.q;j["qd"]=r.qd;j["tau"]=r.tau;
        j["receipt_mono_s"]=r.feedback_mono_s;
    }else if(r.kind==Kind::Inference){j["event"]="policy_inference";j["obs63"]=r.obs;
        j["inferred"]=r.inferred;if(r.inferred){j["raw_action16"]=r.raw_action;j["processed_action16"]=r.processed_action;}
        j["observation_valid"]=r.observation_valid;j["source_age_s"]=r.source_age_s;j["source_deadline_mono_s"]=r.deadline_s;
        j["epoch"]=r.epoch;j["source_seq"]=r.source_seq;j["publish_seq"]=r.publish_seq;
    }else if(r.kind==Kind::PerceptionFault){j["event"]="perception_fault";
        j["state"]=r.state;j["target_mode"]=r.target_mode;j["safe_mode"]=r.safe_mode;
        j["joystick"]=r.joystick;j["imu_rpy_omega_acc"]=r.imu;
        j["damping_reset_seq"]=r.damping_reset_seq;j["source_reason"]=r.source_reason;
        j["observation_valid"]=r.observation_valid;j["source_age_s"]=r.source_age_s;
        j["source_deadline_mono_s"]=r.deadline_s;j["grace_deadline_mono_s"]=r.grace_deadline_s;
        j["epoch"]=r.epoch;j["source_seq"]=r.source_seq;j["publish_seq"]=r.publish_seq;
        j["fault_generation"]=r.fault_generation;j["store_revision"]=r.store_revision;
        j["latest_wire_captured"]=r.wire_captured;
        j["latest_wire"]={{"source_seq",r.wire_source_seq},{"publish_seq",r.wire_publish_seq},
            {"source_stamp_s",r.wire_stamp_s},{"receipt_mono_s",r.wire_receipt_mono_s},
            {"store_revision",r.wire_store_revision},{"epoch",r.wire_epoch},{"reason",r.wire_reason}};
    }else {j["event"]="controller_state";j["state"]=r.state;j["target_mode"]=r.target_mode;
        j["safe_mode"]=r.safe_mode;j["joystick"]=r.joystick;j["imu_rpy_omega_acc"]=r.imu;
        j["damping_reset_seq"]=r.damping_reset_seq;j["remote_command_stamp_s"]=r.remote_command_stamp_s;}
    return j;
}
// SafeController::Start sets safe mode 3 before its judge thread starts. This
// startup wait is normal: mode 0 is first published after stable safety inputs.
// Once armed, any later safety stop remains an incident, even while idle.
// Joint command faults are incidents in every phase.
class StateFaultGate {
    bool armed_=false;
public:
    bool observe(int safe_mode,bool joint_fault=false) {
        if(safe_mode==0)armed_=true;
        return joint_fault || (armed_ && safe_mode!=0);
    }
};
class Recorder {
    struct Cell {std::atomic<std::uint64_t> seq{0};Record record;};
    std::unique_ptr<Cell[]> queue_;
    std::atomic<std::uint64_t> enqueue_{0},dropped_{0};
    std::uint64_t dequeue_=0,written_=0,bytes_=0;
    std::atomic<bool> stopping_{false},failed_{false},sealed_{false};
    std::atomic<bool> has_fault_{false};
    std::atomic<std::uint64_t> recovery_requested_{0},recovery_complete_{0},recovery_barrier_{0};
    std::thread worker_;
    int directory_fd_=-1,lock_fd_=-1,fd_=-1;
    std::size_t segment_=0;
    double last_sync_=0;
    double fault_time_=0;
    std::string model_sha_,session_;
    bool enabled_=false;
    bool append(const std::string&s) {
        auto*p=s.data();std::size_t left=s.size();
        while(left){const auto n=::write(fd_,p,left);if(n<0&&errno==EINTR)continue;if(n<=0)return false;p+=n;left-=n;}
        bytes_+=s.size();return true;
    }
    bool open_segment() {
        if(fd_>=0){if(::fsync(fd_)!=0)return false;::close(fd_);fd_=-1;}
        const auto filename="segment_"+std::to_string(segment_)+".jsonl";
        // Never follow a link or truncate an arbitrary inode supplied by a user.
        fd_=::openat(directory_fd_,filename.c_str(),O_WRONLY|O_CREAT|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK,0600);
        if(fd_<0)return false;
        struct stat st{};
        if(fstat(fd_,&st)!=0 || !S_ISREG(st.st_mode) || st.st_nlink!=1 || st.st_uid!=geteuid())return false;
        if(ftruncate(fd_,0)!=0)return false;
        bytes_=0;
        return append(nlohmann::json({{"event","segment_start"},{"schema",1},{"session",session_},
            {"segment",segment_},{"pid",getpid()},{"wall_s",wall()},{"mono_s",mono()},
            {"ledger_sha256",joint_limits::ledger_sha256},{"model_sha256",model_sha_},
            {"columns",{"kp","position","kd","velocity","torque"}},
            {"coordinate","SDK robot angles; hardware commands include dir/offset"},
            {"dropped",dropped_.load()}}).dump()+"\n");
    }
    bool write_record(const Record&r) {
        auto j=encode(r);j["record_seq"]=written_;j["session"]=session_;
        const auto s=j.dump()+"\n";
        if(bytes_+s.size()>joint_limits::incident_segment_bytes){
            segment_=(segment_+1)%joint_limits::incident_segments;
            if(!open_segment())return false;
        }
        if(!append(s))return false;
        ++written_;
        const auto now=mono();
        if(r.fault && fault_time_==0){
            fault_time_=now;
            // A bounded first-trigger file survives subsequent ring rotation.
            int fault_fd=::openat(directory_fd_,"first_fault.json",O_WRONLY|O_CREAT|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK,0600);
            if(fault_fd<0)return false;
            struct stat st{};
            bool ok=fstat(fault_fd,&st)==0 && S_ISREG(st.st_mode) && st.st_nlink==1 && st.st_uid==geteuid() && ftruncate(fault_fd,0)==0;
            const char*p=s.data();std::size_t left=s.size();
            while(ok && left){const auto n=::write(fault_fd,p,left);if(n<0&&errno==EINTR)continue;
                if(n<=0){ok=false;break;}p+=n;left-=n;}
            if(::fsync(fault_fd)!=0 || ::fsync(directory_fd_)!=0)ok=false;
            ::close(fault_fd);if(!ok)return false;
        }
        if(r.fault || now-last_sync_>=joint_limits::incident_sync_s){
            if(::fsync(fd_)!=0)return false;
            last_sync_=now;
        }
        return true;
    }
    void failed()noexcept {
        if(!failed_.exchange(true))std::fprintf(stderr,"[INCIDENT_RECORDER_FAILED] high-gain commands will be rejected; stopping/damping remains available\n");
    }
    bool reopen_after_recovery(std::uint64_t ticket) {
        // Worker-only disk work. Preserve the complete old ring and trigger
        // before opening a fresh ring; no history is overwritten by R2.
        struct statvfs v{};
        if(fstatvfs(directory_fd_,&v)!=0 || double(v.f_bavail)*v.f_frsize<
           joint_limits::incident_disk_reserve_bytes+joint_limits::incident_segment_bytes*joint_limits::incident_segments)
            return false;
        if(fd_>=0){if(::fsync(fd_)!=0)return false;::close(fd_);fd_=-1;}
        const auto archive="incident_"+session_;
        if(::mkdirat(directory_fd_,archive.c_str(),0700)!=0)return false;
        const int archive_fd=::openat(directory_fd_,archive.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        if(archive_fd<0)return false;
        bool ok=true;
        for(std::size_t i=0;i<=joint_limits::incident_segments;++i){
            const auto name=i==joint_limits::incident_segments?std::string("first_fault.json"):
                "segment_"+std::to_string(i)+".jsonl";
            if(::renameat(directory_fd_,name.c_str(),archive_fd,name.c_str())!=0 && errno!=ENOENT)ok=false;
        }
        if(::fsync(archive_fd)!=0 || ::fsync(directory_fd_)!=0)ok=false;
        ::close(archive_fd);
        if(!ok)return false;
        session_=std::to_string(getpid())+"_"+std::to_string(std::chrono::system_clock::now().time_since_epoch().count())+"_r"+std::to_string(ticket);
        segment_=0;written_=0;fault_time_=0;last_sync_=0;
        if(!open_segment() || ::fsync(fd_)!=0 || ::fsync(directory_fd_)!=0)return false;
        has_fault_=false;sealed_=false;
        return true;
    }
    void work()noexcept {
        setpriority(PRIO_PROCESS,0,10);
        try{
            for(;;){
                const auto ticket=recovery_requested_.load(std::memory_order_acquire);
                if(ticket!=recovery_complete_.load(std::memory_order_relaxed) &&
                   dequeue_>=recovery_barrier_.load(std::memory_order_acquire)){
                    if(failed_ || !reopen_after_recovery(ticket))failed();
                    recovery_complete_.store(ticket,std::memory_order_release);
                }
                if(fault_time_>0 && !sealed_ && mono()-fault_time_>=joint_limits::incident_post_fault_s){
                    const auto s=nlohmann::json({{"event","incident_frozen"},{"session",session_},
                        {"mono_s",mono()},{"reason","first_fault_pre_and_post_history_preserved_restart_required"}}).dump()+"\n";
                    if(bytes_+s.size()>joint_limits::incident_segment_bytes){segment_=(segment_+1)%joint_limits::incident_segments;if(!open_segment())failed();}
                    if(!append(s)||::fsync(fd_)!=0)failed();
                    sealed_=true;
                }
                auto&c=queue_[dequeue_%joint_limits::incident_queue_capacity];
                if(c.seq.load(std::memory_order_acquire)==dequeue_+1){
                    Record r=c.record;c.seq.store(dequeue_+joint_limits::incident_queue_capacity,std::memory_order_release);++dequeue_;
                    if(!sealed_ && !write_record(r))failed();
                }else if(stopping_)break;
                else std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if(fd_>=0){const auto s=nlohmann::json({{"event","session_end"},{"session",session_},{"written",written_},
                {"dropped",dropped_.load()},{"complete",!failed_ && dropped_==0}}).dump()+"\n";
                if(bytes_+s.size()>joint_limits::incident_segment_bytes){segment_=(segment_+1)%joint_limits::incident_segments;if(!open_segment())failed();}
                if(!append(s)||::fsync(fd_)!=0)failed();}
        }catch(...){failed();}
    }
    void close_fds()noexcept {
        for(int* fd:{&fd_,&lock_fd_,&directory_fd_})if(*fd>=0){::close(*fd);*fd=-1;}
    }
public:
    explicit Recorder(const std::string&path) {
        // Empty is used ONLY by policy-only offline fixtures, not hardware.
        if(path.empty())return;
        try{
            directory_fd_=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
            if(directory_fd_<0)throw std::runtime_error("Incident directory missing or symlink");
            struct stat st{};if(fstat(directory_fd_,&st)!=0 || st.st_uid!=geteuid() || (st.st_mode&0022))
                throw std::runtime_error("Incident directory must be owned by controller and not group/world writable");
            lock_fd_=::openat(directory_fd_,".lock",O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC,0600);
            if(lock_fd_<0||flock(lock_fd_,LOCK_EX|LOCK_NB)!=0)throw std::runtime_error("Incident directory already in use");
            struct statvfs v{};
            if(fstatvfs(directory_fd_,&v)!=0 || double(v.f_bavail)*v.f_frsize<
               joint_limits::incident_disk_reserve_bytes+joint_limits::incident_segment_bytes*joint_limits::incident_segments)
                throw std::runtime_error("Incident recording disk reserve unavailable");
            session_=std::to_string(getpid())+"_"+std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            const char* sha=std::getenv("M20_INCIDENT_MODEL_SHA256");model_sha_=sha?sha:"unspecified_offline";
            // Replace the oldest segment on restart, retaining other sessions.
            std::filesystem::file_time_type oldest=std::filesystem::file_time_type::max();
            for(std::size_t i=0;i<joint_limits::incident_segments;++i){
                auto p=std::filesystem::path(path)/("segment_"+std::to_string(i)+".jsonl");
                if(!std::filesystem::exists(p)){segment_=i;break;}
                auto time=std::filesystem::last_write_time(p);if(time<oldest){oldest=time;segment_=i;}
            }
            if(!open_segment() || ::fsync(fd_)!=0 || ::fsync(directory_fd_)!=0)throw std::runtime_error("Cannot initialize incident segment");
            queue_=std::make_unique<Cell[]>(joint_limits::incident_queue_capacity);
            for(std::size_t i=0;i<joint_limits::incident_queue_capacity;++i)queue_[i].seq.store(i);
            enabled_=true;worker_=std::thread([this]{work();});
        }catch(...){close_fds();throw;}
    }
    ~Recorder(){if(enabled_){stopping_=true;if(worker_.joinable())worker_.join();}close_fds();}
    bool healthy()const{return enabled_ && !failed_ && !sealed_;}
    bool frozen()const{return sealed_;}
    bool enabled()const{return enabled_;}
    // Called only after the policy worker has exited and zero-gain/zero-kd
    // output has been sent, while holding the final publisher mutex. Existing
    // queue reservations drain before the worker archives/reopens the ring.
    // An I/O failure is never cleared by R2.
    bool restart_after_explicit_recovery() {
        if(!enabled_ || failed_)return false;
        if(!has_fault_ && !sealed_)return healthy();
        auto ticket=recovery_requested_.load(std::memory_order_acquire);
        if(ticket==recovery_complete_.load(std::memory_order_acquire)){
            recovery_barrier_.store(enqueue_.load(std::memory_order_acquire),std::memory_order_release);
            recovery_requested_.store(++ticket,std::memory_order_release);
        }
        const auto deadline=mono()+2.;
        while(recovery_complete_.load(std::memory_order_acquire)!=ticket && !failed_ && mono()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return recovery_complete_.load(std::memory_order_acquire)==ticket && healthy();
    }
    bool put(const Record&r)noexcept {
        if(!enabled_)return false;
        if(failed_ || sealed_)return false;
        if(recovery_requested_.load(std::memory_order_acquire)!=recovery_complete_.load(std::memory_order_acquire))return false;
        auto position=enqueue_.load(std::memory_order_relaxed);
        for(int attempt=0;attempt<16;++attempt){
            auto&c=queue_[position%joint_limits::incident_queue_capacity];
            const auto sequence=c.seq.load(std::memory_order_acquire);
            const auto difference=static_cast<std::int64_t>(sequence-position);
            if(difference==0){
                if(enqueue_.compare_exchange_weak(position,position+1,std::memory_order_relaxed)){
                    if(r.fault)has_fault_.store(true,std::memory_order_release);
                    c.record=r;c.seq.store(position+1,std::memory_order_release);return true;}
            }else if(difference<0)break;
            else position=enqueue_.load(std::memory_order_relaxed);
        }
        ++dropped_;failed();return false;
    }
    static Recorder& global(){static Recorder r(std::getenv("M20_INCIDENT_DIR")?std::getenv("M20_INCIDENT_DIR"):"");return r;}
};
}}
