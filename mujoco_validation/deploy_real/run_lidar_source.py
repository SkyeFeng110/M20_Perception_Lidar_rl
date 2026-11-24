"""Isolated front LiDAR publisher; never opens a motor interface.

Package layout: this script, contract, bin/{rslidar_network,timestamp guard},
config/config.yaml. Supports raw multicast or private unicast input.
Relayed GOS input is published only on loopback.
Does not change vendor files, services, routes, or interface queue settings.
"""
import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import selectors
import re
import resource
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interface_ipv4(name):
    info = json.loads(subprocess.check_output(['ip', '-j', 'address', 'show', 'dev', name]))
    addresses = [v['local'] for i in info for v in i.get('addr_info', []) if v['family'] == 'inet']
    if len(addresses) != 1 or not info or 'LOWER_UP' not in info[0]['flags']:
        raise ValueError('Expected one IPv4 address on a live interface: '+name)
    return str(ipaddress.IPv4Address(addresses[0]))


def constrain_process(contract):
    cpus = set(contract['cpus'])
    os.sched_setaffinity(0, cpus)
    if os.sched_getaffinity(0) != cpus:
        raise ValueError('Requested sensor CPU allocation failed kernel readback')
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < contract['nice']:
        os.setpriority(os.PRIO_PROCESS, 0, contract['nice'])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-mac', required=True)
    p.add_argument('--expected-serial', required=True)
    p.add_argument('--sensor-interface', required=True)
    p.add_argument('--dds-interface', required=True, choices=['lo'])
    p.add_argument('--relay-input', action='store_true', required=True, help='private GOS unicast ports, local DDS only')
    p.add_argument('--seconds', type=int, required=True, help='bounded probe, or 0 inside board tmux')
    p.add_argument('--output', type=Path, required=True, help='new evidence directory')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    contract = json.loads((root/'lidar_source_contract.json').read_text())
    if os.geteuid() != 0:
        p.error('root required for full duplicate-process and board identity checks')
    serial = Path('/sys/firmware/devicetree/base/serial-number').read_bytes().rstrip(b'\0').decode()
    macs = {f.read_text().strip().lower() for f in Path('/sys/class/net').glob('*/address')}
    if serial != a.expected_serial or a.expected_mac.lower() not in macs:
        p.error('Board identity mismatch; nothing started')
    if not (contract['minimum_probe_seconds'] <= a.seconds <= contract['maximum_probe_seconds'] or
            a.seconds == 0 and os.environ.get('TMUX')):
        p.error('Use bounded probe duration, or board tmux for continuous sensor operation')
    if a.sensor_interface == a.dds_interface:
        p.error('This repair requires distinct sensor and board interfaces')
    sensor_ip, dds_ip = interface_ipv4(a.sensor_interface), interface_ipv4(a.dds_interface)
    files = {'bin/rslidar_network': 'driver_sha256',
             'bin/libm20_lidar_timestamp_guard.so': 'timestamp_guard_sha256',
             'config/config.yaml': 'reference_front_config_sha256'}
    for name, key in files.items():
        if digest(root/name) != contract[key]:
            p.error('Source identity mismatch: '+name)
    binary = root/'bin/rslidar_network'
    for f in Path('/proc').glob('[0-9]*/exe'):
        try:
            target = f.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if target == binary:
            p.error('This sensor publisher is already active')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:')))*1024
    if available < contract['minimum_available_memory_bytes']:
        p.error('Insufficient measured RAM for extra sensor driver and reserve')
    constrain_process(contract)
    if a.output.exists():
        p.error('Output directory already exists')
    # Serialize instances independently of their evidence output paths.
    lock = os.open(root/'.source.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.output.mkdir(mode=0o700, parents=True)
    a.output = a.output.resolve()
    # Derived from the selected live interface, rather than a saved robot address.
    (a.output/'bin').mkdir()
    (a.output/'config').mkdir()
    reference = (root/'config/config.yaml').read_text()
    if reference.count('  - driver:\n') != 1 or 'host_address:' in reference:
        p.error('Expected the pinned single front LiDAR configuration')
    generated_config = a.output/'config/config.yaml'
    generated = reference.replace('  - driver:\n', '  - driver:\n      host_address: '+sensor_ip+'\n')
    if a.relay_input:
        if dds_ip != '127.0.0.1':
            p.error('Relayed source must publish only on loopback')
        generated, n = re.subn(r'(group_address:) [^\n]+', r'\1 '+contract['relay_group'], generated)
        if n != 1:
            p.error('Expected one front multicast group')
        for key, ports in contract['relay_ports'].items():
            generated, n = re.subn(r'('+key+r':) '+str(ports['original'])+r'(?=\s)',
                                   r'\g<1> '+str(ports['relay']), generated)
            if n != 1:
                p.error('Front UDP port contract mismatch: '+key)
    generated_config.write_text(generated)
    profile = a.output/'dds.xml'
    profile.write_text('''<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
<transport_descriptors><transport_descriptor><transport_id>sensor_output</transport_id><type>UDPv4</type>
<interfaceWhiteList><address>'''+dds_ip+'''</address></interfaceWhiteList>
<sendBufferSize>'''+str(contract['dds_socket_buffer_bytes'])+'''</sendBufferSize><receiveBufferSize>'''+str(contract['dds_socket_buffer_bytes'])+'''</receiveBufferSize>
<maxMessageSize>'''+str(contract['dds_max_message_bytes'])+'''</maxMessageSize><non_blocking_send>true</non_blocking_send>
</transport_descriptor></transport_descriptors>
<participant profile_name="sensor_output" is_default_profile="true"><rtps>
<userTransports><transport_id>sensor_output</transport_id></userTransports>
<useBuiltinTransports>false</useBuiltinTransports></rtps></participant></profiles>''')
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(contract['domain']), FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),
               LD_PRELOAD=str(root/'bin/libm20_lidar_timestamp_guard.so'))
    env.pop('M20_LIDAR_TELEMETRY_HOST', None)
    cpus = ','.join(map(str, contract['cpus']))
    command = [str(binary)]  # Inherit the verified parent affinity and nice exactly.
    receipt = dict(started=False, motor_commands=0, board_serial=serial,
                   sensor_interface=a.sensor_interface, sensor_ip=sensor_ip,
                   dds_interface=a.dds_interface, dds_ip=dds_ip, available_memory_bytes=available,
                   contract_sha256=digest(root/'lidar_source_contract.json'),
                   actual_cpus=sorted(os.sched_getaffinity(0)), actual_nice=os.getpriority(os.PRIO_PROCESS,0),
                   source_files={name: digest(root/name) for name in files},
                   generated_config_sha256=digest(generated_config),
                   requested_seconds=a.seconds, acceptance='not established by process lifetime')
    receipt_file = a.output/'source.json'
    receipt_file.write_text(json.dumps(receipt, indent=2)+'\n')
    child = None
    start = time.monotonic()
    log = None
    try:
        child = subprocess.Popen(command, cwd=a.output/'bin', env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock,))
        receipt.update(started=True, pid=child.pid)
        receipt_file.write_text(json.dumps(receipt, indent=2)+'\n')
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            index, written = 0, 0
            log = (a.output/'driver.0.log').open('wb')
            while not stopped and (a.seconds == 0 or time.monotonic()-start < a.seconds):
                if child.poll() is not None:
                    break
                for key, mask in selector.select(.2):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        stopped = True
                        break
                    if written + len(data) > contract['log_segment_bytes']:
                        log.close()
                        index = (index + 1) % contract['log_segments']
                        log = (a.output/('driver.%d.log' % index)).open('wb')
                        written = 0
                    log.write(data)
                    written += len(data)
        receipt['unexpected_child_exit'] = child.poll()
    finally:
        if child is not None:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if child.poll() is not None:
                    break
                os.killpg(child.pid, sig)
                try:
                    child.wait(timeout=contract['stop_grace_seconds'])
                except subprocess.TimeoutExpired:
                    pass
            receipt['child_returncode'] = child.poll()
        if log:
            log.close()
        receipt.update(elapsed_s=time.monotonic()-start, stopped=True)
        receipt_file.write_text(json.dumps(receipt, indent=2)+'\n')
        os.close(lock)
    print(json.dumps(receipt, indent=2))
    return 1 if receipt.get('unexpected_child_exit') is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
