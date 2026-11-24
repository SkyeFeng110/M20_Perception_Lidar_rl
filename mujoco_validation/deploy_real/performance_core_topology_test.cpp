#include "performance_core_topology.hpp"
#include <iostream>
int main(){try{
    using namespace m20::cpu_topology;
    std::map<int,Cpu> cpus;
    for(int i=0;i<4;++i)cpus[i]={405,0x412fd050,true};
    for(int i=4;i<8;++i)cpus[i]={i<6?1024:982,0x414fd0b0,true};
    if(performance_cores(cpus)!=std::set<int>({4,5,6,7}))return 1;
    cpus[7].online=false;
    if(performance_cores(cpus)!=std::set<int>({4,5,6}))return 2;
    cpus[6].midr=0x414fd0b3; // revision change retains architecture class
    if(!performance_cores(cpus).count(6))return 3;
    cpus[6].midr=0;bool rejected=false;
    try{performance_cores(cpus);}catch(...){rejected=true;}
    if(!rejected)return 4;
    cpus[6].midr=0x414fd0b0;for(auto&[id,c]:cpus){(void)id;c.capacity=1024;}
    rejected=false;try{performance_cores(cpus);}catch(...){rejected=true;}
    if(!rejected)return 5;
    std::cout<<"PERFORMANCE_CORE_TOPOLOGY=PASS cases=5 little_and_offline_cores_excluded\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
