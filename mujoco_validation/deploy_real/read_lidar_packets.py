"""Passive Linux packet counts for configured LiDAR UDP ports; no payload saved."""
import argparse
from collections import Counter
import json
import socket
import struct
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',required=True)
    args=parser.parse_args()
    ports={6691,6692,6681,6682,7781,7782}
    counts=Counter();total=0
    with socket.socket(socket.AF_PACKET,socket.SOCK_RAW,socket.htons(3)) as stream:
        stream.bind((args.interface,0));stream.settimeout(.25)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            try:data=stream.recv(2048)
            except socket.timeout:continue
            total+=1
            if len(data)<42 or data[12:14]!=b'\x08\x00' or data[23]!=17:continue
            offset=14+(data[14]&15)*4
            if len(data)<offset+8:continue
            source,destination=struct.unpack_from('!HH',data,offset)
            if destination in ports or source in ports:
                key=socket.inet_ntoa(data[26:30])+':'+str(source)+'->'+str(destination)
                counts[key]+=1
    print(json.dumps(dict(interface=args.interface,seconds=5,total_ethernet_packets=total,
                          lidar_udp_packets=dict(counts),no_packets_transmitted=True)))


if __name__=='__main__':main()
