"""Forward only the selected front LiDAR's UDP payloads to private GOS ports.

No motor interfaces, payload edits, routes, vendor files, or global NIC changes.
"""
import argparse
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time
from run_lidar_source import interface_ipv4, constrain_process


def neighbor_mac(address, interface):
    rows = json.loads(subprocess.check_output(['ip', '-j', 'neigh', 'show', 'dev', interface]))
    matches = [r.get('lladdr', '').lower() for r in rows if r['dst'] == address]
    if len(matches) != 1 or not matches[0]:
        raise ValueError('No independently measurable neighbor MAC for '+address)
    return matches[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-mac', required=True)
    p.add_argument('--expected-serial', required=True)
    p.add_argument('--sensor-interface', required=True)
    p.add_argument('--board-interface', required=True)
    p.add_argument('--source-address', required=True)
    p.add_argument('--source-mac', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--destination-mac', required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    c = json.loads((root/'lidar_source_contract.json').read_text())
    if os.geteuid() != 0:
        p.error('root required for board checks and explicit egress interface')
    serial = Path('/sys/firmware/devicetree/base/serial-number').read_bytes().rstrip(b'\0').decode()
    macs = {f.read_text().strip().lower() for f in Path('/sys/class/net').glob('*/address')}
    if serial != a.expected_serial or a.expected_mac.lower() not in macs:
        p.error('Board identity mismatch; nothing started')
    if not (c['minimum_probe_seconds'] <= a.seconds <= c['maximum_probe_seconds'] or
            a.seconds == 0 and os.environ.get('TMUX')):
        p.error('Bounded duration, or board tmux, required')
    source = str(ipaddress.IPv4Address(a.source_address))
    destination = str(ipaddress.IPv4Address(a.destination))
    sensor_ip, board_ip = interface_ipv4(a.sensor_interface), interface_ipv4(a.board_interface)
    if a.sensor_interface == a.board_interface or source == destination:
        p.error('Distinct sensor and board paths required')
    if (neighbor_mac(source, a.sensor_interface) != a.source_mac.lower() or
            neighbor_mac(destination, a.board_interface) != a.destination_mac.lower()):
        p.error('Sensor/peer MAC mismatch; nothing started')
    if a.output.exists():
        p.error('Evidence directory already exists')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:')))*1024
    if available < c['minimum_available_memory_bytes']:
        p.error('Insufficient measured RAM and reserve')
    constrain_process(c)
    lock = os.open(root/'.relay.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.output.mkdir(mode=0o700, parents=True)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    stats = {key: dict(received=0, forwarded=0, bytes=0, wrong_source=0,
                       send_errors=0, max_input_gap_s=0., last_input_mono_s=None)
             for key in c['relay_ports']}
    result = dict(motor_commands=0, source_address=source, destination=destination,
                  source_mac=a.source_mac, destination_mac=a.destination_mac,
                  sensor_interface=a.sensor_interface, board_interface=a.board_interface,
                  available_memory_bytes=available, ports=stats, payloads_unchanged=True,
                  acceptance='packet forwarding only; does not establish perception validity')
    output = a.output/'relay.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    sockets = []
    start = time.monotonic()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender, selectors.DefaultSelector() as selector:
            sender.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, a.board_interface.encode()+b'\0')
            sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, c['dds_socket_buffer_bytes'])
            sender.bind((board_ip, 0))
            sender.setblocking(False)
            for key, ports in c['relay_ports'].items():
                receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sockets.append(receiver)
                receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, c['dds_socket_buffer_bytes'])
                receiver.bind((c['front_group'], ports['original']))
                receiver.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                    socket.inet_aton(c['front_group'])+socket.inet_aton(sensor_ip))
                receiver.setblocking(False)
                selector.register(receiver, selectors.EVENT_READ, key)
            while not stopped and (a.seconds == 0 or time.monotonic()-start < a.seconds):
                for event, mask in selector.select(.2):
                    data, address = event.fileobj.recvfrom(65535)
                    item = stats[event.data]
                    if address[0] != source:
                        item['wrong_source'] += 1
                        continue
                    now = time.monotonic()
                    previous = item['last_input_mono_s']
                    if previous is not None:
                        item['max_input_gap_s'] = max(item['max_input_gap_s'], now-previous)
                    item['last_input_mono_s'] = now
                    item['received'] += 1
                    try:
                        count = sender.sendto(data, (destination, c['relay_ports'][event.data]['relay']))
                        if count != len(data):
                            raise OSError('short UDP send')
                        item['forwarded'] += 1
                        item['bytes'] += count
                    except OSError:
                        item['send_errors'] += 1
    finally:
        for receiver in sockets:
            receiver.close()
        os.close(lock)
        result.update(elapsed_s=time.monotonic()-start, stopped=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if all(s['forwarded'] and not s['send_errors'] for s in stats.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
