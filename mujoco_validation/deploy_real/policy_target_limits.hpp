#pragma once
#include <array>
namespace m20::target_limits {
inline constexpr char ledger_sha256[]="0b44290faab286fa4f2a6d4117cfbb741220a3b40ed63cc22dac90e313090138";
inline constexpr std::array<double,3> margin_rad={0.03,0.05,0.08};
inline constexpr std::array<double,3> max_rate_rad_s={50.0,50.0,50.0};
}
