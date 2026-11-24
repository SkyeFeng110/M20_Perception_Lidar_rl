#pragma once
// Schema 2 carries current source health independently of retained geometry.
#include "staircase_packet.hpp"
#include <limits>
namespace m20::v11 {
using Json=nlohmann::json;
using Feature=std::array<float,6>;
struct BundleView {
 bool valid=false; Feature features{};
 m20::v11::SourceIdentity source{},health_source{};
 double health_deadline=0,content_deadline=0,geometry_deadline=0;
 FeatureSelection selection=FeatureSelection::Invalid;
};
class ObservationBundle {
 m20::staircase::PacketGate health_;
 double timeout_,geometry_deadline_=0,expiry_stamp_=0;
 uint64_t geometry_seq_=0;double geometry_stamp_=0;
 Feature geometry_{};m20::v11::SourceIdentity geometry_id_{};
 bool admitted_=false,has_geometry_=false;
 bool recoverable_unavailable_=false;
 std::string error_="waiting";
 uint64_t fault_publish_=0,fault_source_=0;
 double fault_stamp_=0;
 std::string fault_epoch_;
 bool RecordSourceFault(const Json& current,double wall,double mono){
  for(const char* key:{"valid","policy_input","source_valid","processing_ok","fixture"})
   if(!current.at(key).is_boolean()||current.at(key).get<bool>())return false;
  if(current.at("observation_status")!="invalid"||current.at("features")!=Json(Feature{}))return false;
  Json shape=current;
  for(const char* key:{"valid","policy_input","source_valid","processing_ok"})shape[key]=true;
  shape["observation_status"]="no_stairs_detected";
  m20::staircase::PacketGate fault_shape;
  const double stamp=current.at("source_stamp_s");
  if(!std::isfinite(wall)||!std::isfinite(mono)||stamp>wall+.05||
     !fault_shape.Accept(shape.dump(),"v11_signed_edges_v2",stamp,mono,timeout_))return false;
  const auto& id=fault_shape.Identity();
  const auto& previous=health_.Identity();
  const std::string epoch=id.epoch.data();
  if((previous.source_seq&&epoch!=previous.epoch.data())||(!fault_epoch_.empty()&&epoch!=fault_epoch_))
   throw std::runtime_error("source_restart_requires_explicit_reset");
  const auto previous_source=std::max(previous.source_seq,fault_source_);
  const double previous_stamp=std::max(previous.source_stamp_s,fault_stamp_);
  if(id.publish_seq<=std::max(previous.publish_seq,fault_publish_)||id.source_seq<previous_source||
     (id.source_seq==previous_source&&stamp!=previous_stamp)||
     (id.source_seq>previous_source&&stamp<=previous_stamp))return false;
  fault_publish_=id.publish_seq;fault_source_=id.source_seq;fault_stamp_=stamp;fault_epoch_=epoch;
  return true;
 }
public:
 explicit ObservationBundle(double timeout=.3):timeout_(timeout){
  if(!std::isfinite(timeout)||timeout<=0||timeout>.3)throw std::invalid_argument("timeout");
 }
 bool Accept(const Json&bundle,double wall,double mono){
  admitted_=false;has_geometry_=false;recoverable_unavailable_=false;
  try{
   if(!bundle.at("schema").is_number_integer()||bundle.at("schema")!=2)throw std::runtime_error("schema");
   const auto&current=bundle.at("current");
   if(!current.at("source_valid").get<bool>()||!current.at("processing_ok").get<bool>()||current.at("fixture").get<bool>())
   {
    const bool authentic_fault=RecordSourceFault(current,wall,mono);
    // Only a structurally valid, same-session unavailability event can enter
    // grace. Format, identity, fixture and contract faults stay hard failures.
    recoverable_unavailable_=authentic_fault && current.value("reason",std::string{})=="source_timeout";
    throw std::runtime_error("source_invalid");
   }
   auto next_health=health_;
   if(!next_health.Accept(current.dump(),"v11_signed_edges_v2",wall,mono,timeout_))
    throw std::runtime_error(next_health.Reason());
   if(fault_publish_){
    const auto& id=next_health.Identity();
    if(fault_epoch_!=id.epoch.data())throw std::runtime_error("source_restart_requires_explicit_reset");
    if(id.publish_seq<=fault_publish_||id.source_seq<=fault_source_||id.source_stamp_s<=fault_stamp_)
     throw std::runtime_error("observation_precedes_source_fault");
   }
   if(health_.Identity().source_seq==next_health.Identity().source_seq&&
      health_.Identity().source_seq!=0&&health_.Feature()!=next_health.Feature())
    throw std::runtime_error("current_measurement_changed");
   const bool positive=next_health.Feature()[0]!=0;
   if(bundle.contains("retained")&&!bundle.at("retained").is_null()){
    const auto&r=bundle.at("retained");const auto&p=r.at("packet");
    const auto seq=p.at("source_seq").get<uint64_t>();const double stamp=p.at("source_stamp_s");
    const double expiry=r.at("expires_stamp_s");
    if(fault_source_&&seq<=fault_source_)throw std::runtime_error("geometry_precedes_source_fault");
    if(!std::isfinite(expiry)||expiry<stamp||expiry>stamp+timeout_+1e-7||
       p.at("epoch")!=current.at("epoch")||p.at("publish_seq")!=current.at("publish_seq")||
       seq>current.at("source_seq").get<uint64_t>()||stamp>current.at("source_stamp_s").get<double>()||
       (seq==current.at("source_seq").get<uint64_t>()&&stamp!=current.at("source_stamp_s").get<double>()))
     throw std::runtime_error("geometry_identity_or_lease");
    // Validate geometry structurally with the real packet parser. Actual age
    // is enforced by the separate immutable monotonic lease below.
    m20::staircase::PacketGate shape;
    if(!shape.Accept(p.dump(),"v11_signed_edges_v2",stamp,mono,timeout_)||shape.Feature()[0]!=1||
       !p.at("source_valid").get<bool>()||!p.at("processing_ok").get<bool>()||p.at("fixture").get<bool>())
     throw std::runtime_error("geometry_shape");
    if((seq==current.at("source_seq").get<uint64_t>()&&shape.Feature()!=next_health.Feature())||
       (positive&&(seq!=current.at("source_seq").get<uint64_t>()||shape.Feature()!=next_health.Feature())))
     throw std::runtime_error("latest_positive_must_replace");
    if(seq<=health_.Identity().source_seq&&seq!=geometry_seq_)
     throw std::runtime_error("unseen_retained_precedes_current");
    if(seq<geometry_seq_||stamp<geometry_stamp_||
       (seq==geometry_seq_&&(stamp!=geometry_stamp_||shape.Feature()!=geometry_||expiry>expiry_stamp_+1e-7))||
       (seq>geometry_seq_&&stamp<=geometry_stamp_))throw std::runtime_error("geometry_reordered_or_renewed");
    const double deadline=mono+std::min(expiry-wall,timeout_-std::max(0.,wall-stamp));
    geometry_deadline_=seq==geometry_seq_?std::min(geometry_deadline_,deadline):deadline;
    geometry_seq_=seq;geometry_stamp_=stamp;expiry_stamp_=expiry;geometry_=shape.Feature();geometry_id_=shape.Identity();
    has_geometry_=true;
   }else if(positive)throw std::runtime_error("positive_missing_lease");
   health_=next_health;admitted_=true;error_="ok";return true;
  }catch(const std::exception&e){error_=e.what();return false;}
 }
 BundleView Read(double mono,double read_to_commit_budget=0)const{
  BundleView out;
  const double use_until=mono+read_to_commit_budget;
  if(!admitted_||!std::isfinite(mono)||!std::isfinite(read_to_commit_budget)||!std::isfinite(use_until)||
     read_to_commit_budget<0||!health_.Valid(mono)||!health_.Valid(use_until))return out;
  out.valid=true;out.health_source=health_.Identity();out.source=out.health_source;
  out.health_deadline=mono+health_.Remaining(mono);out.content_deadline=out.health_deadline;
  out.geometry_deadline=has_geometry_?geometry_deadline_:0;
  if(has_geometry_&&mono+read_to_commit_budget<geometry_deadline_){
   out.features=geometry_;out.source=geometry_id_;out.content_deadline=std::min(out.health_deadline,geometry_deadline_);out.selection=FeatureSelection::Retained;
  }else {
   // Chosen semantics A: no admissible geometric feature -> zero actor input.
   // ExpiredZero is a derived policy fallback, NOT a measured flat/no-stairs
   // claim. Its provenance remains distinct from an actual CurrentZero.
   out.selection=has_geometry_?FeatureSelection::ExpiredZero:FeatureSelection::CurrentZero;
  }
  return out;
 }
 const std::string&Error()const{return error_;}
 bool HasSourceEvent()const{return health_.Identity().source_seq!=0||fault_publish_!=0;}
 bool RecoverableUnavailable()const{return recoverable_unavailable_;}
};
}
