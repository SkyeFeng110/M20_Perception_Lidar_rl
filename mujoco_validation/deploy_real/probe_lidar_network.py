"""Bounded passive Ethernet/IPv4/UDP summary; never saves packet payloads."""
import argparse
from collections import Counter
import json
import socket
import struct
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interface', required=True)
    p.add_argument('--seconds', type=float, default=5)
    a = p.parse_args()
    if not 0 < a.seconds <= 30:
        p.error('seconds must be in (0, 30]')
    counts, sources, udp, ports = Counter(), Counter(), Counter(), Counter()
    with socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3)) as s:
        s.bind((a.interface, 0))
        s.settimeout(.2)
        start = time.monotonic()
        while time.monotonic() - start < a.seconds:
            try:
                data, address = s.recvfrom(65535)
            except socket.timeout:
                continue
            counts['ethernet_packets'] += 1
            if len(data) < 14:
                continue
            sources[data[6:12].hex(':')] += 1
            offset, kind = 14, struct.unpack_from('!H', data, 12)[0]
            while kind in (0x8100, 0x88a8) and len(data) >= offset + 4:
                kind = struct.unpack_from('!H', data, offset + 2)[0]
                offset += 4
            counts['ethertype_%04x' % kind] += 1
            if kind != 0x0800 or len(data) < offset + 20:
                continue
            if data[offset + 9] != 17:
                continue
            header = (data[offset] & 15) * 4
            if header < 20 or len(data) < offset + header + 8:
                continue
            # Non-first fragments do not contain UDP ports.
            if struct.unpack_from('!H', data, offset + 6)[0] & 0x1fff:
                counts['udp_nonfirst_fragments'] += 1
                continue
            src = socket.inet_ntoa(data[offset + 12:offset + 16])
            dst = socket.inet_ntoa(data[offset + 16:offset + 20])
            sport, dport = struct.unpack_from('!HH', data, offset + header)
            ports[str(dport)] += 1
            direction = 'outgoing' if address[2] == socket.PACKET_OUTGOING else 'incoming'
            udp['%s %s:%d -> %s:%d' % (direction, src, sport, dst, dport)] += 1
        stats = struct.unpack('II', s.getsockopt(263, 6, 12)[:8])
    print(json.dumps(dict(interface=a.interface, elapsed_s=time.monotonic()-start,
                         counts=dict(counts), ethernet_sources=dict(sources),
                         udp_flows=dict(udp.most_common(80)), udp_destination_ports=dict(ports),
                         socket_packets=stats[0], socket_drops=stats[1],
                         no_packets_transmitted=True), indent=2))


if __name__ == '__main__':
    main()
