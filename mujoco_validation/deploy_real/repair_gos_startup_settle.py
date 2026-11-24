"""Bound the ROS startup thread-set check using a release resource budget.

Keeps thread identity, affinity/priority readback, rollback, and failure refusal.
No sensor/actuator algorithms or motor interfaces are run by this installer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from release_utils import refresh_checksums, replace_once

EXPECTED_SOURCE = '9e4b743ad9aefbf9e3781dbb118a4f5bd238836a1fc51b5a13bd0a5c0c4a5f3a'


def repair(root, expected_mac, expected_serial):
    serial = Path('/sys/firmware/devicetree/base/serial-number').read_bytes().rstrip(b'\0').decode()
    macs = {p.read_text().strip().lower() for p in Path('/sys/class/net').glob('*/address')}
    if os.geteuid() != 0 or serial != expected_serial or expected_mac.lower() not in macs:
        raise ValueError('GOS root/identity check failed; nothing changed')
    if subprocess.run(['pgrep', '-x', 'm20_deploy'], stdout=subprocess.DEVNULL).returncode == 0:
        raise ValueError('Stop this release supervisor before installation')
    subprocess.run(['sha256sum', '-c', '--quiet', 'RELEASE_SHA256SUMS'], cwd=root, check=True)
    source = root/'native_tools/gos_resources.hpp'
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_SOURCE:
        raise ValueError('Unreviewed resource implementation')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:')))*1024
    if available <= 5*1024**3 or shutil.disk_usage(root).free <= 512*1024**2:
        raise ValueError('Reference compiler budget unavailable: 4 GiB plus 1 GiB reserve')
    out = Path(tempfile.mkdtemp(prefix='startup_settle_build_', dir=root/'artifacts'))
    shutil.copytree(root/'native_tools', out, dirs_exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parent/'performance_core_topology_test.cpp', out/'performance_core_topology_test.cpp')
    text = replace_once(source.read_text(), 'for(int pass=0;pass<5;++pass){',
                        'for(int pass=0;pass<profile.value("thread_settle_passes",5);++pass){')
    text = replace_once(text, 'require(seen==roles,"All four perception roles required");',
                        'require(seen==roles,"All four perception roles required");\n'
                        ' if(j.contains("thread_settle_passes"))require(j.at("thread_settle_passes").is_number_integer()&&'
                        'j.at("thread_settle_passes").get<int>()>=5&&j.at("thread_settle_passes").get<int>()<=100,'
                        '"Bounded startup settle budget required");')
    (out/'gos_resources.hpp').write_text(text)
    compiler = ['nice', '-n', '19', 'taskset', '-c', '4,5', 'prlimit', '--as=4294967296', '--',
                'g++', '-std=c++17', '-O2', '-pthread', '-I', str(out)]
    subprocess.run(compiler+[str(out/'performance_core_topology_test.cpp'), '-o', str(out/'topology_test')], check=True)
    subprocess.run([str(out/'topology_test')], check=True)
    subprocess.run(compiler+[str(out/'gos_only_main.cpp'), '-ldl', '-o', str(out/'m20_deploy')], check=True)
    help_text = subprocess.check_output([str(out/'m20_deploy'), '--help'], text=True)
    if 'gos-run|stop|check-session' not in help_text or 'aos-run' in help_text:
        raise ValueError('GOS-only entry validation failed')
    profile = root/'runtime/gos_resources.json'
    config = json.loads(profile.read_text())
    # Existing per-pass sleep remains unchanged. This gives a bounded startup
    # interval rather than rejecting a normal wave of lazy DDS/TBB threads.
    config['thread_settle_passes'] = 100
    for name in ('native_tools/m20_deploy', 'native_tools/gos_resources.hpp', 'runtime/gos_resources.json'):
        original = root/name
        backup = original.with_name(original.name+'.before_startup_settle.disabled')
        if backup.exists():
            raise ValueError('Startup repair backup already exists')
    for name in ('native_tools/m20_deploy', 'native_tools/gos_resources.hpp', 'runtime/gos_resources.json'):
        original = root/name
        backup = original.with_name(original.name+'.before_startup_settle.disabled')
        shutil.copy2(original, backup)
        backup.chmod(0o444)
    for name in ('gos_resources.hpp', 'm20_deploy'):
        temporary = root/'native_tools'/(name+'.next')
        shutil.copy2(out/name, temporary)
        os.replace(temporary, root/'native_tools'/name)
    temporary = profile.with_name(profile.name+'.next')
    temporary.write_text(json.dumps(config, indent=2)+'\n')
    os.replace(temporary, profile)
    refresh_checksums(root)
    receipt = dict(motor_commands=0, available_memory_bytes=available,
                   change='Bounded startup settling; all identity/resource/failure checks retained',
                   roles_and_quiet_units_unchanged=True, thread_settle_passes=config['thread_settle_passes'],
                   elf_sha256=hashlib.sha256((root/'native_tools/m20_deploy').read_bytes()).hexdigest())
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('release', type=Path)
    p.add_argument('--expected-mac', required=True)
    p.add_argument('--expected-serial', required=True)
    a = p.parse_args()
    repair(a.release.resolve(), a.expected_mac, a.expected_serial)
