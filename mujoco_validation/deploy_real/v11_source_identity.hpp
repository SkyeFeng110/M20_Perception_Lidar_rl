#pragma once
#include <array>
#include <cstdint>
#include <cmath>
#include <algorithm>
namespace m20 { namespace v11 {
// Copied under the same lock as the six features; never reconstructed from a
// later callback. Steady times are local to AOS, stamp is the GOS source clock.
struct SourceIdentity {
    std::array<char,129> epoch{};
    std::uint64_t source_seq=0, publish_seq=0;
    double source_stamp_s=0, receipt_mono_s=0;
};
enum class FeatureSelection { Invalid, CurrentZero, Retained, ExpiredZero, DegradedHold };
enum class LeaseStatus { Ready, GeometryExpired, Fault };
struct ObservationLease {
    int wire_schema = 0;
    FeatureSelection selection = FeatureSelection::Invalid;
    SourceIdentity health_source{};
    double selection_mono_s = 0, selection_budget_s = 0;
    double health_deadline_s = 0, geometry_deadline_s = 0;
    // Separate authorization for bounded degraded operation; never fresh health.
    double grace_deadline_s = 0;
    std::uint64_t fault_generation = 0, store_revision = 0;
};
struct CommitRequest {
    ObservationLease selected{};
    double content_deadline_s = 0;
    bool observation_valid = false;
    std::uint64_t admitted_fault_generation = 0;
};
struct CommitResult {
    LeaseStatus status = LeaseStatus::Fault;
    bool committed = false;
    std::uint64_t store_revision = 0, fault_generation = 0, admitted_fault_generation = 0;
    double guard_mono_s = 0;
};
inline LeaseStatus ClassifyLease(const ObservationLease& lease, double content_deadline,
                                 bool valid, double now) {
    if (!valid || !std::isfinite(now)) return LeaseStatus::Fault;
    if (lease.wire_schema != 2) {
        if (lease.wire_schema != 0 && lease.wire_schema != 1) return LeaseStatus::Fault;
        return content_deadline == 0 ||
            (std::isfinite(content_deadline) && now <= content_deadline)
            ? LeaseStatus::Ready : LeaseStatus::Fault;
    }
    if (!std::isfinite(lease.selection_mono_s) || lease.selection_mono_s <= 0 ||
        now < lease.selection_mono_s || !std::isfinite(lease.health_deadline_s) ||
        !std::isfinite(content_deadline) ||
        !std::isfinite(lease.selection_budget_s) || lease.selection_budget_s < 0 ||
        !std::isfinite(lease.geometry_deadline_s)) return LeaseStatus::Fault;
    if (!std::isfinite(lease.grace_deadline_s) || lease.grace_deadline_s < 0)
        return LeaseStatus::Fault;
    if (lease.selection == FeatureSelection::DegradedHold) {
        return lease.grace_deadline_s >= lease.health_deadline_s &&
            content_deadline == lease.grace_deadline_s &&
            lease.selection_mono_s + lease.selection_budget_s < content_deadline &&
            now < content_deadline ? LeaseStatus::Ready : LeaseStatus::Fault;
    }
    if (lease.selection_mono_s + lease.selection_budget_s >= lease.health_deadline_s)
        return LeaseStatus::Fault;
    if (lease.selection!=FeatureSelection::Retained && lease.selection!=FeatureSelection::CurrentZero &&
        lease.selection!=FeatureSelection::ExpiredZero) return LeaseStatus::Fault;
    if (now >= lease.health_deadline_s)
        return lease.grace_deadline_s > now ? LeaseStatus::GeometryExpired : LeaseStatus::Fault;
    if (lease.selection == FeatureSelection::Retained) {
        if (content_deadline != std::min(lease.geometry_deadline_s, lease.health_deadline_s) ||
            content_deadline <= lease.selection_mono_s ||
            lease.geometry_deadline_s <= lease.selection_mono_s + lease.selection_budget_s)
            return LeaseStatus::Fault;
        return now < content_deadline ? LeaseStatus::Ready : LeaseStatus::GeometryExpired;
    }
    if (content_deadline != lease.health_deadline_s) return LeaseStatus::Fault;
    if (lease.selection == FeatureSelection::CurrentZero)
        return lease.geometry_deadline_s == 0 ? LeaseStatus::Ready : LeaseStatus::Fault;
    if (lease.selection == FeatureSelection::ExpiredZero &&
        lease.geometry_deadline_s <= lease.selection_mono_s + lease.selection_budget_s)
        return LeaseStatus::Ready;
    return LeaseStatus::Fault;
}
}}
