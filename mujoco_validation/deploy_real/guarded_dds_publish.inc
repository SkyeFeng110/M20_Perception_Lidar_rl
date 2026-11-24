    // The only position/velocity/torque motor-control publisher. Both generic
    // states and lease-guarded policy commands terminate here.
    m20::incident::StateFaultGate incident_state_fault_gate_;
    bool PublishJointMessage(const drdds::msg::JointsDataCmd& proposed,
                             const m20::v11::CommitRequest* source_request = nullptr,
                             m20::v11::CommitResult* source_result = nullptr) {
        std::lock_guard<std::mutex> lock(joint_publish_mutex_);
        auto& recorder=m20::incident::Recorder::global();
        m20::joint_safety::Command command{};
        m20::joint_safety::Feedback feedback;
        {
            std::lock_guard<std::mutex> state_lock(joint_state_mutex_);
            feedback.receipt_mono_s=joint_receipt_mono_s_;
            for(int i=0;i<16;++i){feedback.q[i]=joint_pos_(i);feedback.qd[i]=joint_vel_(i);feedback.tau[i]=joint_tau_(i);}
        }
        bool calibration_ok=true,is_passive=true;
        for(int i=0;i<16;++i){
            const auto&j=proposed.data.joints_data[i];
            is_passive=is_passive && j.kp==0 && std::isfinite(j.kd) && j.kd>=0 && j.velocity==0 && j.torque==0;
            calibration_ok=calibration_ok && std::isfinite(joint_config_[i].offset) &&
                std::isfinite(joint_config_[i].dir) && std::abs(joint_config_[i].dir)==1;
            command[i]={j.kp,j.position*joint_config_[i].dir+joint_config_[i].offset,
                        j.kd,j.velocity*joint_config_[i].dir,j.torque*joint_config_[i].dir};
        }
        if(is_passive)for(auto&j:command){j[0]=0;j[1]=0;j[3]=0;j[4]=0;}
        const double now=m20::incident::mono();
        auto result=(!is_passive && !calibration_ok)?
            joint_guard_.fail(m20::joint_safety::Reason::Calibration):
            joint_guard_.validate(command,feedback,now,recorder.healthy());
        if(result.reason==m20::joint_safety::Reason::Latched && !joint_command_fault_.load(std::memory_order_acquire))
            result=joint_guard_.first_fault();
        m20::incident::Record event;
        event.kind=m20::incident::Kind::Command;event.mono_s=now;event.wall_s=m20::incident::wall();
        event.inference_id=m20::alignment::context().inference_id;
        event.feedback_mono_s=feedback.receipt_mono_s;event.q=feedback.q;event.qd=feedback.qd;event.tau=feedback.tau;
        event.accepted=result.ok();event.latched=joint_guard_.latched();event.joint=result.joint;
        event.violation_value=result.value;event.violation_limit=result.limit;
        event.fault=!result.ok() && result.reason!=m20::joint_safety::Reason::Latched;
        m20::incident::reason(event,m20::joint_safety::name(result.reason));
        for(int i=0;i<16;++i){const auto&j=proposed.data.joints_data[i];
            event.dir[i]=joint_config_[i].dir;event.offset[i]=joint_config_[i].offset;
            for(int k=0;k<5;++k)event.sdk_command[i*5+k]=command[i][k];
            event.hardware_command[i*5]=j.kp;event.hardware_command[i*5+1]=j.position;
            event.hardware_command[i*5+2]=j.kd;event.hardware_command[i*5+3]=j.velocity;event.hardware_command[i*5+4]=j.torque;
        }
        // Queue the candidate before issuing torque. If evidence cannot be
        // queued, reject high gain; stopping/damping still goes out immediately.
        if(!recorder.put(event) && !is_passive && result.ok())
            result=joint_guard_.fail(m20::joint_safety::Reason::Recorder);
        auto outgoing=proposed;
        if(!result.ok() || is_passive){
            // Canonical zero-position/velocity/feedforward message: never let an
            // invalid unused position, stale feedback or bad calibration block
            // a reverse action. Preserve requested passive kd if valid.
            command=result.ok()?command:m20::joint_safety::damping();
            for(int i=0;i<16;++i){auto&j=outgoing.data.joints_data[i];
                j.position=0;j.velocity=0;j.torque=0;j.kp=0;j.kd=command[i][2];j.control_word=kIndexMotorControl;
                command[i][0]=0;command[i][1]=0;command[i][3]=0;command[i][4]=0;
            }
        }
        if(!result.ok()){
            const bool first=!joint_command_fault_.exchange(true,std::memory_order_acq_rel);
            if(first)std::fprintf(stderr,"[JOINT_COMMAND_FAULT] reason=%s joint=%d value=%.9g limit=%.9g; damping latched until explicit R2 recovery\n",
                m20::joint_safety::name(result.reason),result.joint,result.value,result.limit);
        }
        double publish_begin=0;
        m20::v11::CommitResult final_commit;
        try{
            if(source_request && result.ok()){
                // Check the exact selected source while holding the publish
                // lock, after target validation and before the actual DDS send.
                // A source can expire while another state owns this lock.
                final_commit=v11_store_.GuardedCommit(*source_request,V11SteadyNow,
                    [&](const m20::v11::CommitResult&){publish_begin=m20::alignment::mono();EmitJointMessage(outgoing);});
                if(source_result)*source_result=final_commit;
                if(!final_commit.committed){
                    event.accepted=false;event.mono_s=m20::incident::mono();
                    m20::incident::reason(event,final_commit.status==m20::v11::LeaseStatus::GeometryExpired?
                        "source_expired_not_published":"source_fault_not_published");
                    recorder.put(event);
                    return false; // no target history update and no joint latch
                }
            }else {publish_begin=m20::alignment::mono();EmitJointMessage(outgoing);}
        }catch(...){
            joint_guard_.fail(m20::joint_safety::Reason::Publisher);
            joint_command_fault_.store(true,std::memory_order_release);
            event.accepted=false;event.latched=true;event.fault=true;
            m20::incident::reason(event,"publisher_exception");recorder.put(event);
            std::fprintf(stderr,"[JOINT_COMMAND_FAULT] publisher_exception; attempting damping\n");
            // Keep a best-effort damping attempt even if the publisher threw.
            for(int i=0;i<16;++i){auto&j=outgoing.data.joints_data[i];j.kp=0;j.position=0;j.velocity=0;j.torque=0;j.kd=i%4==3?1.f:2.f;}
            try{EmitJointMessage(outgoing);}catch(...){}
            return false;
        }
        const double publish_end=m20::alignment::mono();
        joint_guard_.committed(command,publish_end);
        for(int i=0;i<16;++i)for(int k=0;k<5;++k)joint_cmd_(i,k)=command[i][k];
        event.accepted=result.ok();event.latched=joint_guard_.latched();
        event.fault=!result.ok() && result.reason!=m20::joint_safety::Reason::Latched;
        event.mono_s=publish_end;event.wall_s=m20::incident::wall();
        m20::incident::reason(event,result.ok()?"published":"rejected_published_damping");
        for(int i=0;i<16;++i){const auto&j=outgoing.data.joints_data[i];
            event.hardware_command[i*5]=j.kp;event.hardware_command[i*5+1]=j.position;
            event.hardware_command[i*5+2]=j.kd;event.hardware_command[i*5+3]=j.velocity;event.hardware_command[i*5+4]=j.torque;}
        if(!recorder.put(event) && !is_passive && result.ok()){
            joint_guard_.fail(m20::joint_safety::Reason::Recorder);joint_command_fault_.store(true,std::memory_order_release);
            for(int i=0;i<16;++i){auto&j=outgoing.data.joints_data[i];j.kp=0;j.position=0;j.velocity=0;j.torque=0;j.kd=i%4==3?1.f:2.f;}
            try{EmitJointMessage(outgoing);}catch(...){} // logging must never block damping
            joint_cmd_.setZero();for(int i=0;i<16;++i)joint_cmd_(i,2)=i%4==3?1.f:2.f;
            return false;
        }
        auto&journal=m20::alignment::Journal::global();
        if(journal.enabled()){
            m20::alignment::Event e;e.kind=m20::alignment::Kind::Publish;
            e.mono_s=publish_end;e.wall_s=m20::alignment::wall();
            e.publish_begin_mono_s=publish_begin;e.publish_duration_s=publish_end-publish_begin;
            e.inference_id=m20::alignment::context().inference_id;e.source=m20::alignment::context().source;
            if(source_request && result.ok()){e.guard=final_commit;e.policy_guarded=true;e.admitted_fault_generation=final_commit.admitted_fault_generation;}
            for(int i=0;i<16;++i){const auto&j=outgoing.data.joints_data[i];
                e.hardware_dir[i]=joint_config_[i].dir;e.hardware_offset[i]=joint_config_[i].offset;
                e.command[i*5]=j.kp;e.command[i*5+1]=j.position;e.command[i*5+2]=j.kd;e.command[i*5+3]=j.velocity;e.command[i*5+4]=j.torque;}
            journal.put(e);
        }
        return result.ok();
    }
    void RejectCommandShape() {
        {std::lock_guard<std::mutex> lock(joint_publish_mutex_);joint_guard_.fail(m20::joint_safety::Reason::Shape);}
        auto msg=drdds::msg::JointsDataCmd();
        // Force the normal rejection path so fault state and evidence agree.
        for(int i=0;i<16;++i){msg.data.joints_data[i].kp=1;msg.data.joints_data[i].control_word=kIndexMotorControl;}
        PublishJointMessage(msg);
    }
public:
    bool HasJointCommandFault() const override {return joint_command_fault_.load(std::memory_order_acquire);}
    bool ResetJointCommandFaultForRecovery() override {
        // R2 recovery is consumed only after the FSM has entered damping.
        // Publish zero gain and zero damping first, then clear the guard and
        // its target history so manual repositioning cannot inherit a stale
        // policy target.
        MatXf passive_cmd = MatXf::Zero(dof_num_, 5);
        if (!PublishJointMessage(MakeJointCommand(passive_cmd))) return false;
        {
            std::lock_guard<std::mutex> lock(joint_publish_mutex_);
            if (!m20::incident::Recorder::global().restart_after_explicit_recovery()) {
                std::cerr << "[RECOVERY] Joints unloaded, but incident recorder could not rearm; fault remains latched.\n";
                return false;
            }
            joint_guard_.reset_for_recovery();
            joint_command_fault_.store(false, std::memory_order_release);
            joint_cmd_.setZero();
        }
        std::cout << "[RECOVERY] Joint command guard reset after passive R2 release" << std::endl;
        return true;
    }
    MatXf GetJointCommand() override {std::lock_guard<std::mutex> lock(joint_publish_mutex_);return joint_cmd_;}
    void RecordPerceptionFault(int state,const UserCommand& uc,const char* trigger) override {
        m20::incident::Record r;r.kind=m20::incident::Kind::PerceptionFault;
        r.fault=true;r.mono_s=m20::incident::mono();r.wall_s=m20::incident::wall();r.state=state;
        r.target_mode=uc.target_mode.load();r.safe_mode=uc.safe_control_mode.load();
        r.damping_reset_seq=uc.damping_reset_seq.load();r.latched=HasJointCommandFault();
        r.inference_id=m20::alignment::context().inference_id;
        const auto s=v11_store_.Read(r.mono_s);
        r.observation_valid=s.valid;r.source_age_s=s.age_seconds;r.deadline_s=s.steady_deadline_seconds;
        r.grace_deadline_s=s.lease.grace_deadline_s;r.fault_generation=s.lease.fault_generation;
        r.store_revision=s.lease.store_revision;r.source_seq=s.source.source_seq;r.publish_seq=s.source.publish_seq;
        std::strncpy(r.epoch,s.source.epoch.data(),sizeof(r.epoch)-1);
        std::strncpy(r.source_reason,s.reason.c_str(),sizeof(r.source_reason)-1);
        m20::incident::reason(r,trigger);
        // Capture the latest packet separately from the failed selected lease.
        // Revisions expose a callback racing this diagnostic snapshot.
        std_msgs::msg::String::SharedPtr wire;
        {std::lock_guard<std::mutex> lock(incident_wire_mutex_);wire=incident_last_wire_;
            r.wire_receipt_mono_s=incident_wire_mono_;r.wire_store_revision=incident_wire_revision_;}
        if(wire&&wire->data.size()<=16384)try{
            const auto envelope=nlohmann::json::parse(wire->data);
            const auto& current=envelope.value("schema",0)==2?envelope.at("current"):envelope;
            r.wire_source_seq=current.value("source_seq",std::uint64_t(0));
            r.wire_publish_seq=current.value("publish_seq",std::uint64_t(0));
            r.wire_stamp_s=current.value("source_stamp_s",0.);
            const auto epoch=current.value("epoch",std::string{}),reason=current.value("reason",std::string{});
            std::strncpy(r.wire_epoch,epoch.c_str(),sizeof(r.wire_epoch)-1);
            std::strncpy(r.wire_reason,reason.c_str(),sizeof(r.wire_reason)-1);r.wire_captured=true;
        }catch(...){/* malformed input still freezes the first fault */}
        r.joystick={uc.forward_vel_scale.load(),uc.side_vel_scale.load(),uc.turnning_vel_scale.load()};
        {std::lock_guard<std::mutex> lock(imu_state_mutex_);for(int i=0;i<3;++i){r.imu[i]=rpy_(i);r.imu[3+i]=omega_body_(i);r.imu[6+i]=acc_(i);}}
        m20::incident::Recorder::global().put(r);
    }
    void RecordControllerState(int state,const UserCommand& uc) override {
        m20::incident::Record r;r.kind=m20::incident::Kind::State;
        r.mono_s=m20::incident::mono();r.wall_s=m20::incident::wall();r.state=state;
        r.target_mode=uc.target_mode.load();r.safe_mode=uc.safe_control_mode.load();
        r.damping_reset_seq=uc.damping_reset_seq.load();r.remote_command_stamp_s=uc.time_stamp.load();
        r.latched=HasJointCommandFault();
        r.fault=incident_state_fault_gate_.observe(r.safe_mode,r.latched);
        if(r.fault)m20::incident::reason(r,r.latched?"joint_command_fault_latched":"sdk_or_remote_safety_stop");
        else if(r.safe_mode!=0)m20::incident::reason(r,"startup_safety_wait");
        r.joystick={uc.forward_vel_scale.load(),uc.side_vel_scale.load(),uc.turnning_vel_scale.load()};
        {std::lock_guard<std::mutex> lock(imu_state_mutex_);for(int i=0;i<3;++i){r.imu[i]=rpy_(i);r.imu[3+i]=omega_body_(i);r.imu[6+i]=acc_(i);}}
        m20::incident::Recorder::global().put(r);
    }
    void SetJointCommand(Eigen::Matrix<float,Eigen::Dynamic,5> input) override {
        if(input.rows()!=16 || input.cols()!=5){RejectCommandShape();return;}
        PublishJointMessage(MakeJointCommand(input));
    }
    m20::v11::CommitResult SetPolicyJointCommand(Eigen::Matrix<float,Eigen::Dynamic,5> input,
            const m20::v11::CommitRequest& request) override {
        if(input.rows()!=16 || input.cols()!=5){RejectCommandShape();return {};}
        const auto msg=MakeJointCommand(input);
        m20::v11::CommitResult result;result.admitted_fault_generation=request.admitted_fault_generation;
        const bool actual_committed=PublishJointMessage(msg,&request,&result);
        if(result.committed && !actual_committed){result.committed=false;result.status=m20::v11::LeaseStatus::Fault;}
        return result;
    }
