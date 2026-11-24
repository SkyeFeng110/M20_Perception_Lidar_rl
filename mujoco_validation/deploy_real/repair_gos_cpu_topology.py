"""Compile/test the GOS-only supervisor with measured ARM core-class checking.

No motor code is run. Does not alter affinity profiles, quiet services or sensor
parameters. The executable is replaced only after its no-hardware tests pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from release_utils import refresh_checksums,replace_once

HERE=Path(__file__).resolve().parent
EXPECTED='54786aebb12ec7c459f74910b82304b566b775ec3fd9c1dc4d7bb017b1daa8fb'


def repair(root,expected_mac,expected_serial):
    serial=Path('/sys/firmware/devicetree/base/serial-number').read_bytes().rstrip(b'\0').decode()
    macs={p.read_text().strip().lower() for p in Path('/sys/class/net').glob('*/address')}
    if serial!=expected_serial or expected_mac.lower() not in macs:raise ValueError('GOS hardware identity mismatch')
    if subprocess.run(['pgrep','-x','m20_deploy'],stdout=subprocess.DEVNULL).returncode==0:
        raise ValueError('A native supervisor is active on GOS')
    subprocess.run(['sha256sum','-c','--quiet','RELEASE_SHA256SUMS'],cwd=root,check=True)
    original=root/'native_tools/gos_resources.hpp'
    if hashlib.sha256(original.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Unreviewed GOS resource source')
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:exe=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe==root/'native_tools/m20_deploy':raise ValueError('Native GOS supervisor already active')
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    # Same reference build budget: 4 GiB per compiler + 1 GiB reserve, one job.
    if available<=5*1024**3 or shutil.disk_usage(root).free<=512*1024**2:raise ValueError('GOS compiler resource reserve unavailable')
    out=Path(tempfile.mkdtemp(prefix='cpu_topology_build_20261118_',dir=root/'artifacts'))
    shutil.copytree(root/'native_tools',out,dirs_exist_ok=True)
    for name in ('performance_core_topology.hpp','performance_core_topology_test.cpp'):
        shutil.copy2(HERE/name,out/name)
    p=out/'gos_resources.hpp'
    text=replace_once(p.read_text(),'#include <climits>','#include <climits>\n#include "performance_core_topology.hpp"')
    start=text.index('inline void gos_validate_big_cpus(');end=text.index('inline Json gos_apply_resources(',start)
    text=text[:start]+'''inline void gos_validate_big_cpus(const Json& profile){
 std::map<int,m20::cpu_topology::Cpu> cpus;
 for(auto& f:fs::directory_iterator("/sys/devices/system/cpu")){
  auto name=f.path().filename().string();if(name.rfind("cpu",0)!=0||name.size()==3||!std::all_of(name.begin()+3,name.end(),[](unsigned char c){return isdigit(c);}))continue;
  if(!fs::exists(f.path()/"cpu_capacity"))continue;
  int id=std::stoi(name.substr(3));bool online=!fs::exists(f.path()/"online")||trim(read(f.path()/"online"))=="1";
  require(fs::exists(f.path()/"regs/identification/midr_el1"),"Missing ARM core identity");
  cpus[id]={std::stoi(read(f.path()/"cpu_capacity")),std::stoull(trim(read(f.path()/"regs/identification/midr_el1")),nullptr,0),online};
 }
 auto big=m20::cpu_topology::performance_cores(cpus);
 auto allowed=trim(read("/sys/fs/cgroup/cpuset/cpuset.cpus"));require(!allowed.empty(),"Read effective CPU allocation");
 for(const auto& v:profile.at("roles"))for(const auto& c:v.at("cpus")){int cpu=c;require(big.count(cpu),"Requested CPU is not an online measured performance-core model");}
}
'''+text[end:]
    p.write_text(text)
    compiler=['nice','-n','19','taskset','-c','4,5','prlimit','--as=4294967296','--','g++','-std=c++17','-O2','-pthread','-I',str(out)]
    subprocess.run(compiler+[str(out/'performance_core_topology_test.cpp'),'-o',str(out/'performance_core_topology_test')],check=True)
    subprocess.run([str(out/'performance_core_topology_test')],check=True)
    # The old release retained the reference main.cpp although its ELF was
    # built from the separate GOS-only entry. Save the actual entry explicitly.
    deployed_help=subprocess.check_output([str(root/'native_tools/m20_deploy'),'--help'],text=True)
    if 'gos-run|stop|check-session' not in deployed_help or 'aos-run' in deployed_help:raise ValueError('Deployed tool is not the GOS-only CLI')
    shutil.copy2(HERE/'gos_main.cpp',out/'gos_only_main.cpp')
    main=(out/'gos_only_main.cpp').read_text()
    if 'GOS_DURABLE_SESSION' not in main or 'else if(mode=="aos-run")' in main:raise ValueError('GOS-only entry required')
    subprocess.run(compiler+[str(out/'gos_only_main.cpp'),'-ldl','-o',str(out/'m20_deploy')],check=True)
    subprocess.run([str(out/'m20_deploy'),'--help'],check=True)
    backup=root/'native_tools/m20_deploy.before_cpu_topology.disabled'
    if backup.exists():raise ValueError('Prior topology backup exists')
    shutil.copy2(root/'native_tools/m20_deploy',backup);backup.chmod(0o444)
    for name in ('gos_resources.hpp','performance_core_topology.hpp','gos_only_main.cpp','m20_deploy'):
        temp=root/'native_tools'/(name+'.next');shutil.copy2(out/name,temp);os.replace(temp,root/'native_tools'/name)
    refresh_checksums(root)
    receipt={'status':'compiled_and_no_hardware_tested','change':'performance core identity plus measured capacity; no capacity-equality assumption',
             'profiles_changed':False,'vendor_services_changed':False,'motor_commands':0,
             'elf_sha256':hashlib.sha256((root/'native_tools/m20_deploy').read_bytes()).hexdigest()}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path)
    p.add_argument('--expected-mac',required=True);p.add_argument('--expected-serial',required=True)
    a=p.parse_args();repair(a.release.resolve(),a.expected_mac,a.expected_serial)
