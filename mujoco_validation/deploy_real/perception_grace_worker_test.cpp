// Actual actor and RL worker; all hardware/UDP factories compiled out.
#define main unused_legacy_fsm_main
#include "staircase_fsm_no_motor_test.cpp"
#undef main
#include "perception_grace_limits.hpp"

static json source(double stamp,unsigned seq,unsigned pub,float h) {
    return {{"schema",1},{"source","gos_staircase"},{"contract","v11_signed_edges_v2"},
        {"layout","B1,signedH1,D1,B2,signedH2,D2"},{"frame","gos_odom"},
        {"epoch","occlusion-worker"},{"source_seq",seq},{"publish_seq",pub},
        {"source_stamp_s",stamp},{"features",h==0?json::array({0.,0.,0.,0.,0.,0.}):json::array({1.,h,.5,0.,0.,0.})},
        {"source_valid",true},{"processing_ok",true},{"valid",true},{"policy_input",true},
        {"observation_status",h==0?"no_stairs_detected":"stairs_detected"},
        {"tracker_applied",true},{"fixture",false}};
}
static json packet(const json& p) {
    return {{"schema",2},{"current",p},{"retained",p["features"][0]==1?
        json{{"packet",p},{"expires_stamp_s",p["source_stamp_s"].get<double>()+.2}}:json(nullptr)}};
}
int main(int argc,char**argv){try{
    if(argc!=3)throw std::runtime_error("usage: perception_grace_worker_test CASE MODEL.onnx");
    const std::string name=argv[1];
    require(name=="ascent"||name=="descent"||name=="flat"||name=="recovery"||name=="hard_fault"||name=="yaw_ramp"||name=="target_limits","unknown case");
    const float height=name=="flat"?0.f:name=="descent"?-.16f:.16f;
    setenv("M20_POLICY_PATH",argv[2],1);setenv("M20_V11_FEATURE_CONTRACT","v11_signed_edges_v2",1);
    rclcpp::init(0,nullptr);
    const json input={{"robot",{{"joint_pos",std::array<float,16>{}},{"joint_vel",std::array<float,16>{}},
        {"rpy",std::array<float,3>{}},{"omega",std::array<float,3>{}}}},{"recorded_6d",std::array<float,6>{}}};
    auto robot=std::make_shared<CaptureRobot>(input);
    robot->bundle_store=std::make_shared<m20::v11::ObservationStore>(m20::v11::ObservationSource::Staircase,.3,
        m20::perception_limits::loss_timeout_s);
    auto user=std::make_shared<FixtureUser>();user->usr_cmd_->forward_vel_scale=.2f;
    auto data=std::make_shared<ControllerData>();data->ri_ptr=robot;data->uc_ptr=user;
    auto control=std::make_shared<qw::RLControlState>(RobotName::M20,"occlusion_worker",data);
    unsigned seq=0,pub=0;
    auto put=[&](float h){const auto p=source(m20::alignment::wall(),++seq,++pub,h);
        require(robot->bundle_store->UpdatePacket(packet(p).dump(),"v11_signed_edges_v2",m20::alignment::wall(),now(),true),"fresh source");};
    put(height);control->OnEnter();
    struct Cleanup{std::shared_ptr<qw::RLControlState> c;~Cleanup(){c->OnExit();}}cleanup{control};
    auto run=[&](double seconds){const auto stop=now()+seconds;
        while(now()<stop){control->Run();std::this_thread::sleep_for(std::chrono::milliseconds(5));}};
    // Allow actor initialization before assigning the source used for the test.
    run(.06);put(height);run(.04);put(height);
    if(name=="target_limits"){
        run(.35);control->OnExit();
        const auto submitted=robot->snapshot();require(submitted.size()>5,"not enough actual worker publications");
        VecXf last=VecXf::Zero(16);
        for(const auto& row:submitted)for(int j=0;j<16;++j)if(j%4!=3){
            const float lo=m20::joint_limits::lower(j)+m20::target_limits::margin_rad[j%4];
            const float hi=m20::joint_limits::upper(j)-m20::target_limits::margin_rad[j%4];
            require(row.command(j,1)>=lo-1e-6&&row.command(j,1)<=hi+1e-6,"actual worker angle projection");
            require(std::abs(row.command(j,1)-last(j))<=m20::target_limits::max_rate_rad_s[j%4]*.005+1e-6,"actual worker publication slew");
            last(j)=row.command(j,1);
        }
        require(control->LastObservationAfterExitForDiagnostics()(6)==.2f,"joystick mapping unexpectedly changed");
        require(!data->perception_fault_latched.load(),"shaping caused source fault");
        std::cout<<"POLICY_TARGET_WORKER=PASS publications="<<submitted.size()<<" scope=actual_actor_no_motors_ideal_feedback\n";
        rclcpp::shutdown();return 0;
    }
    if(name=="yaw_ramp"){
        user->usr_cmd_->forward_vel_scale=1.0f;
        user->usr_cmd_->side_vel_scale=.5f;
        user->usr_cmd_->turnning_vel_scale=-.6f;
        run(.12);
        const float early=control->AppliedYawForDiagnostics();
        require(std::abs(early+.6f)<1e-5,"full stick yaw was delayed at actor input");
        control->OnExit();
        require(control->LastObservationAfterExitForDiagnostics()(6)==1.0f &&
                control->LastObservationAfterExitForDiagnostics()(7)==.5f,"full stick translation was delayed at actor input");
        control->OnEnter();
        require(std::abs(control->AppliedYawForDiagnostics()+.6f)<1e-5,"full turning magnitude lost");
        user->usr_cmd_->turnning_vel_scale=0;run(.06);control->OnExit();
        require(control->LastObservationAfterExitForDiagnostics()(8)==0,"neutral yaw delayed in actor");
        require(!data->perception_fault_latched.load(),"ramp caused a source stop");
        std::cout<<"YAW_ACTOR_WORKER=PASS early_yaw="<<early<<" scope=actual_actor_no_motors_ideal_feedback\n";
        rclcpp::shutdown();return 0;
    }
    const double stamp_mono=now(),deadline=stamp_mono+m20::perception_limits::loss_timeout_s;
    run(.35);const auto before=robot->snapshot().size();
    require(!data->perception_fault_latched.load(),"short loss must not latch");
    require(robot->perception_fault_records.load()==0,"short loss must not freeze incident history");
    require(robot->ReadV11Observation().lease.selection==m20::v11::FeatureSelection::DegradedHold,"worker must actually enter degraded hold");
    user->usr_cmd_->forward_vel_scale=.35f;
    run(.12);control->OnExit();
    const auto observation=control->LastObservationAfterExitForDiagnostics();
    require(observation.size()==63&&observation(6)==.35f,"live commands must still reach the actor during loss");
    require(observation(58)==height,"signed terrain stays in actual actor input");
    const auto during=robot->snapshot();require(during.size()>before,"actor keeps producing commands during loss");
    for(const auto& command:during)require(!command.command.col(0).isZero(),"no damping during short occlusion");
    control->OnEnter();
    if(name=="hard_fault"){
        require(!robot->bundle_store->UpdatePacket("{}","v11_signed_edges_v2",m20::alignment::wall(),now(),true),"malformed packet");
        run(.04);require(data->perception_fault_latched.load(),"hard fault must stop immediately");
        control->OnExit();
        require(robot->perception_fault_records.load()==1,"hard fault must record exactly once");
    }else if(name=="recovery"){
        put(-height);run(.1);control->OnExit();
        require(!data->perception_fault_latched.load(),"fresh recovery must stay in policy");
        require(robot->perception_fault_records.load()==0,"fresh recovery must not freeze incident history");
        require(control->LastObservationAfterExitForDiagnostics()(58)==-height,"recovery updates actual actor terrain");
    }else{
        run(std::max(0.,deadline-.05-now()));
        require(!data->perception_fault_latched.load(),"policy must continue up to the long-loss deadline");
        run(.11);
        require(data->perception_fault_latched.load(),"long loss must stop");
        require(robot->perception_fault_records.load()==1,"worker stop must record first perception fault exactly once");
        const auto after=robot->snapshot();require(after.back().command.col(0).isZero(),"long loss commands damping");
        for(const auto& command:after)
            if(!command.command.col(0).isZero())require(command.mono<=deadline+.01,"no active command beyond loss deadline");
    }
    control->OnExit();rclcpp::shutdown();
    std::cout<<"PERCEPTION_GRACE_WORKER=PASS case="<<name<<" active_short_loss_commands="<<during.size()-before
        <<" scope=actual_actor_worker_no_motors_ideal_feedback\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
