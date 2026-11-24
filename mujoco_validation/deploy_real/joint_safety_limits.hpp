#pragma once
#include <array>
#include <cstdint>
namespace m20 { namespace joint_limits {
inline constexpr const char* ledger_sha256 = "4ec1de04636847ee9cbad789cba598a017e70512efc150ea05557015c1929dec";
inline constexpr bool operator_limits_confirmed = true;
inline constexpr std::array<double,3> front_left_lower_rad = {-0.436,-2.286,-2.809};
inline constexpr std::array<double,3> front_left_upper_rad = {0.611,2.583,2.792};
inline constexpr std::array<double,3> leg_velocity_rad_s = {10,12,15};
inline constexpr std::array<double,3> leg_torque_nm = {32.4,76.4,76.4};
inline constexpr std::array<double,3> target_rate_rad_s = {50,50,50};
inline constexpr double control_tick_s = 0.005;
inline constexpr double policy_period_s = 0.02;
inline constexpr double feedback_timeout_s = 0.05;
inline constexpr std::uint64_t incident_segment_bytes = 16777216;
inline constexpr std::uint64_t incident_segments = 4;
inline constexpr std::uint64_t incident_disk_reserve_bytes = 536870912;
inline constexpr double incident_sync_s = 1.0;
inline constexpr double incident_post_fault_s = 2.0;
inline constexpr std::uint64_t incident_queue_capacity = 1024;
inline constexpr std::array<float,12> effective_lower = {-0.436f,-2.286f,-2.809f,-0.611f,-2.286f,-2.809f,-0.436f,-2.583f,-2.792f,-0.611f,-2.583f,-2.792f};
inline constexpr std::array<float,12> effective_upper = {0.611f,2.583f,2.792f,0.436f,2.583f,2.792f,0.611f,2.286f,2.809f,0.436f,2.286f,2.809f};
inline constexpr std::array<double,16> tracking_error_max_rad = {0.65,0.0,0.0,0.0,0.65,0.0,0.0,0.0,0.65,0.0,0.0,0.0,0.65,0.0,0.0,0.0};
inline constexpr std::array<double,16> tracking_error_proxy_nm = {52.0,95.0,85.0,0.0,52.0,95.0,85.0,0.0,52.0,95.0,85.0,0.0,52.0,95.0,85.0,0.0};
inline constexpr float lower(int i) { return effective_lower[(i/4)*3+i%4]; }
inline constexpr float upper(int i) { return effective_upper[(i/4)*3+i%4]; }
}}
