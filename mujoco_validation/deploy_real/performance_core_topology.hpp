#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
namespace m20 { namespace cpu_topology {
struct Cpu {int capacity=0;std::uint64_t midr=0;bool online=true;};
inline std::set<int> performance_cores(const std::map<int,Cpu>& cpus){
    int lowest=0x7fffffff,highest=0;std::uint64_t fastest_model=0;
    // Compare ARM implementer+part, allowing different revisions and OPP bins.
    constexpr std::uint64_t model_mask=0xff00fff0ULL;
    for(const auto&[id,c]:cpus){(void)id;if(!c.online)continue;
        if(c.capacity<=0 || (c.midr&model_mask)==0)throw std::runtime_error("Incomplete CPU identity/capacity evidence");
        lowest=std::min(lowest,c.capacity);
        if(c.capacity>highest){highest=c.capacity;fastest_model=c.midr&model_mask;}
    }
    if(highest<=lowest)throw std::runtime_error("Heterogeneous measured CPU topology required");
    std::set<int> result;
    for(const auto&[id,c]:cpus)
        if(c.online && c.capacity>lowest && (c.midr&model_mask)==fastest_model)result.insert(id);
    if(result.empty())throw std::runtime_error("No measured performance core class");
    return result;
}
}}
