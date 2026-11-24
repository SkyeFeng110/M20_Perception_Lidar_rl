#pragma once
#include "v11_observation_bundle.hpp"
#include <limits>
#include <mutex>
#include <stdexcept>

namespace m20 { namespace v11 {
enum class ObservationSource { Legacy, Staircase };
inline ObservationSource ParseObservationSource(const std::string& value) {
    if (value == "legacy") return ObservationSource::Legacy;
    if (value == "staircase") return ObservationSource::Staircase;
    throw std::invalid_argument("M20_V11_OBSERVATION_SOURCE must be legacy or staircase");
}
// Hardware deployment selection, separate from the reusable legacy/sim store.
inline ObservationSource HardwareObservationSource(const char* value) {
#ifdef M20_GOS_ONLY_CONTROL
    if (value && std::string(value) != "staircase")
        throw std::invalid_argument("GOS-only controller rejects non-GOS observation sources");
    return ObservationSource::Staircase;
#else
    return ParseObservationSource(value ? value : "legacy");
#endif
}
struct ObservationSnapshot {
    ObservationLease lease{};
    SourceIdentity source{};
    SourceIdentity health_source{};
    FeatureSelection selection = FeatureSelection::Invalid;
    std::array<float, 6> features{};
    bool valid = false;
    float age_seconds = std::numeric_limits<float>::infinity();
    double steady_deadline_seconds = 0;
    double health_deadline_seconds = 0;
    double geometry_deadline_seconds = 0;
    std::string reason = "waiting_for_source";
};
// Shared by the DDS callbacks and the no-hardware replay. Selection is fixed
// for this object's lifetime; there is no runtime fallback to another source.
class ObservationStore {
public:
    explicit ObservationStore(ObservationSource source, double timeout = .3, double loss_timeout = 0)
        : source_(source), timeout_(timeout), bundle_(std::min(timeout, .3)), loss_timeout_(loss_timeout) {
        if (!std::isfinite(timeout) || timeout <= 0 || timeout > .5)
            throw std::invalid_argument("invalid V11 source timeout");
        if (!std::isfinite(loss_timeout) || loss_timeout < 0 ||
            (loss_timeout > 0 && loss_timeout < timeout))
            throw std::invalid_argument("invalid V11 perception loss timeout");
    }
    ObservationSource Source() const { return source_; }
    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++fault_generation_;
        ++store_revision_;
        gate_ = staircase::PacketGate{};
        bundle_ = ObservationBundle(std::min(timeout_, .3));
        wire_schema_ = 0;
        features_ = {};
        valid_ = admitted_ = restart_latched_ = false;
        last_mono_ = 0;
        reason_ = "waiting_for_source_after_reset";
        grace_allowed_=false;cached_={};grace_deadline_=0;
    }
    bool UpdateLegacy(const std::array<float, 6>& features, bool valid, double mono) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (source_ != ObservationSource::Legacy) return false;
        valid_ = valid && std::isfinite(mono);
        for (float v : features) valid_ = valid_ && std::isfinite(v);
        features_ = features; last_mono_ = mono;
        reason_ = valid_ ? "ok" : "invalid_legacy_observation";
        return valid_;
    }
    bool UpdatePacket(const std::string& wire, const std::string& contract,
                      double wall, double mono, bool unique_publisher,
                      std::uint64_t* update_revision = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++store_revision_;
        if (update_revision) *update_revision = store_revision_;
        if (source_ != ObservationSource::Staircase) return false;
        struct FaultOnReject {
            bool& admitted;
            bool& grace_allowed;
            std::uint64_t& generation;
            ~FaultOnReject() { if (!admitted && !grace_allowed) ++generation; }
        } fault_on_reject{admitted_, grace_allowed_, fault_generation_};
        // Recovery after the hard loss deadline must cancel the running
        // episode even when the control loop never sampled the expired gap.
        if (wire_schema_==2 && grace_allowed_ && !CanHold(mono,0))
            ++fault_generation_;
        else if (wire_schema_ == 2 && admitted_ && !bundle_.Read(mono).valid)
            if (!CanHold(mono,0)) ++fault_generation_;
        admitted_ = false;
        const bool previous_grace=grace_allowed_;
        grace_allowed_=false;
        if (restart_latched_) { reason_ = "source_restart_requires_explicit_reset"; return false; }
        if (!unique_publisher) { reason_ = "publisher_not_unique"; return false; }
        if (wire.size() > 16384) { reason_ = "oversized_packet"; return false; }
        // Explicitly exclude preview/fixture traffic from production admission.
        try {
            auto p = nlohmann::json::parse(wire);
            if (!p.at("schema").is_number_integer()) {
                reason_ = "malformed_schema"; return false;
            }
            const int schema = p.at("schema").get<int>();
            if (wire_schema_ && schema != wire_schema_) {
                reason_ = "wire_schema_changed"; return false;
            }
            if (schema == 2) {
                if (contract != "v11_signed_edges_v2") {
                    reason_ = "bundle_contract_mismatch"; return false;
                }
                admitted_ = bundle_.Accept(p, wall, mono);
                reason_ = bundle_.Error();
                restart_latched_ = reason_ == "source_restart_requires_explicit_reset";
                if (admitted_ || bundle_.HasSourceEvent()) wire_schema_ = schema;
                if (admitted_) RememberGood(mono);
                else if (previous_grace && bundle_.RecoverableUnavailable() && CanHoldCached(mono,0))
                    grace_allowed_=true;
                return admitted_;
            }
            if (!p.at("source_valid").is_boolean() ||
                !p.at("source_valid").get<bool>() ||
                !p.at("fixture").is_boolean() || p.at("fixture").get<bool>()) {
                reason_ = "preview_or_fixture"; return false;
            }
        } catch (const std::exception&) { reason_ = "malformed_packet"; return false; }
        admitted_ = gate_.Accept(wire, contract, wall, mono, timeout_);
        if (admitted_) wire_schema_ = 1;
        reason_ = gate_.Reason();
        restart_latched_ = reason_ == "source_restart_requires_explicit_reset";
        return admitted_;
    }
    ObservationSnapshot Read(double mono, double read_to_commit_budget = 0) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ReadLocked(mono, read_to_commit_budget);
    }
    template<class Clock, class Publish>
    CommitResult GuardedCommit(const CommitRequest& request, Clock&& clock, Publish&& publish) {
        std::lock_guard<std::mutex> lock(mutex_);
        CommitResult result;
        result.admitted_fault_generation = request.admitted_fault_generation;
        const auto latest = ReadLocked(clock(), 0);
        result.store_revision = store_revision_;
        result.fault_generation = fault_generation_;
        result.guard_mono_s = clock();
        result.status = ClassifyLease(request.selected, request.content_deadline_s,
                                      request.observation_valid, result.guard_mono_s);
        if (ClassifyLease(latest.lease, latest.steady_deadline_seconds, latest.valid,
                          result.guard_mono_s) == LeaseStatus::Fault ||
            request.selected.fault_generation != request.admitted_fault_generation ||
            fault_generation_ != request.admitted_fault_generation)
            result.status = LeaseStatus::Fault;
        if (result.status == LeaseStatus::Ready) {
            publish(result);
            result.committed = true;
        }
        return result;
    }
private:
    ObservationSnapshot ReadLocked(double mono, double read_to_commit_budget) const {
        auto result=ReadStrictLocked(mono,read_to_commit_budget);
        // A latest positive measurement may exhaust its short geometry lease
        // before the source health lease. Enter the same explicit degraded
        // mode then, avoiding a false zero followed by restored old stairs.
        const bool latest_geometry_expired=result.valid &&
            result.selection==FeatureSelection::ExpiredZero && cached_.features[0]!=0 &&
            cached_.source.source_seq==cached_.health_source.source_seq;
        if ((!result.valid || latest_geometry_expired) && CanHold(mono,read_to_commit_budget)) {
            result=cached_;
            result.valid=true;result.selection=FeatureSelection::DegradedHold;
            result.reason="short_perception_loss_hold";
            result.age_seconds=static_cast<float>(std::max(0.,mono-(grace_deadline_-loss_timeout_)));
            result.steady_deadline_seconds=grace_deadline_;
            result.lease.selection=result.selection;
        }
        result.lease.store_revision=store_revision_;
        result.lease.fault_generation=fault_generation_;
        result.lease.selection_mono_s=mono;
        result.lease.selection_budget_s=read_to_commit_budget;
        result.lease.grace_deadline_s=grace_allowed_?grace_deadline_:0;
        return result;
    }
    bool CanHoldCached(double mono,double budget)const {
        return loss_timeout_>0 && cached_.valid && std::isfinite(mono) &&
            std::isfinite(budget) && budget>=0 && mono>=cached_.lease.selection_mono_s &&
            std::isfinite(mono+budget) && mono+budget<grace_deadline_;
    }
    bool CanHold(double mono,double budget)const {
        return grace_allowed_ && CanHoldCached(mono,budget);
    }
    void RememberGood(double mono) {
        if (loss_timeout_<=0 || wire_schema_!=2) return;
        const auto next=ReadStrictLocked(mono,0);
        if (!next.valid) return;
        // Only genuinely new measured source frames refresh the loss budget.
        if (!cached_.valid || next.health_source.source_seq>cached_.health_source.source_seq) {
            cached_=next;
            if (next.source.source_seq!=next.health_source.source_seq) {
                cached_.features={};cached_.source=next.health_source;
                cached_.selection=FeatureSelection::CurrentZero;
                cached_.geometry_deadline_seconds=0;
                cached_.lease.geometry_deadline_s=0;
            }
            grace_deadline_=next.health_deadline_seconds+loss_timeout_-std::min(timeout_,.3);
        }
        grace_allowed_=CanHoldCached(mono,0);
    }
    ObservationSnapshot ReadStrictLocked(double mono, double read_to_commit_budget) const {
        ObservationSnapshot result;
        result.lease.store_revision = store_revision_;
        result.lease.wire_schema = wire_schema_;
        result.lease.selection_mono_s = mono;
        result.lease.selection_budget_s = read_to_commit_budget;
        result.lease.fault_generation = fault_generation_;
        result.reason = reason_;
        if (source_ == ObservationSource::Staircase) {
            if (wire_schema_ == 2) {
                const auto view = bundle_.Read(mono, read_to_commit_budget);
                result.valid = admitted_ && view.valid;
                if (result.valid) {
                    result.features = view.features;
                    result.source = view.source;
                    result.health_source = view.health_source;
                    result.selection = view.selection;
                    result.steady_deadline_seconds = view.content_deadline;
                    result.health_deadline_seconds = view.health_deadline;
                    result.geometry_deadline_seconds = view.geometry_deadline;
                    result.lease.selection = view.selection;
                    result.lease.health_source = view.health_source;
                    result.lease.health_deadline_s = view.health_deadline;
                    result.lease.geometry_deadline_s = view.geometry_deadline;
                    result.age_seconds = static_cast<float>(std::max(0.,
                        std::min(timeout_, .3) - (view.health_deadline - mono)));
                }
                if (!result.valid && result.reason == "ok") result.reason = "source_timeout";
                return result;
            }
            result.valid = admitted_ && gate_.Valid(mono);
            if (result.valid) {
                result.features = gate_.Feature();
                result.source = gate_.Identity();
                result.age_seconds = static_cast<float>(std::max(0., timeout_ - gate_.Remaining(mono)));
                result.steady_deadline_seconds = mono + gate_.Remaining(mono);
            }
        } else {
            const double age = mono - last_mono_;
            result.valid = valid_ && std::isfinite(age) && age >= 0 && age <= timeout_;
            if (result.valid) {
                result.features = features_; result.age_seconds = age;
                result.steady_deadline_seconds = last_mono_ + timeout_;
            }
        }
        if (!result.valid && result.reason == "ok") result.reason = "source_timeout";
        return result;
    }
private:
    const ObservationSource source_;
    const double timeout_;
    mutable std::mutex mutex_;
    staircase::PacketGate gate_;
    ObservationBundle bundle_;
    int wire_schema_ = 0;
    std::uint64_t fault_generation_ = 0, store_revision_ = 0;
    std::array<float, 6> features_{};
    bool valid_ = false, admitted_ = false, restart_latched_ = false;
    double last_mono_ = 0;
    std::string reason_ = "waiting_for_source";
    const double loss_timeout_;
    bool grace_allowed_=false;
    double grace_deadline_=0;
    ObservationSnapshot cached_{};
};
}}  // namespace m20::v11
