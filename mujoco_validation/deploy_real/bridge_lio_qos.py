"""GOS-local BEST_EFFORT -> RELIABLE sensor bridge for the saved LIO.

Republishes the same messages, including their original headers and point
timestamps. No motor API, geometry, clock correction, or freshness relaxation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from run_lidar_source import constrain_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    c = json.loads((root/'lidar_source_contract.json').read_text())
    if not (c['minimum_probe_seconds'] <= a.seconds <= c['maximum_probe_seconds'] or
            a.seconds == 0 and os.environ.get('TMUX')) or a.output.exists():
        p.error('Bounded duration or board tmux, and a new output path, required')
    constrain_process(c)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Imu, PointCloud2
    # The private input topics have only the local LIO consumer. Existing checks
    # still determine whether the original data are valid and fresh.
    os.environ['ROS_DOMAIN_ID'] = str(c['domain'])
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    rclpy.init(args=[])
    node = rclpy.create_node('m20_lio_sensor_qos_bridge')
    depth = c['qos_bridge_depth']
    incoming = QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST,
                          reliability=ReliabilityPolicy.BEST_EFFORT)
    outgoing = QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST,
                          reliability=ReliabilityPolicy.RELIABLE)
    counts, pubs, subs = {}, [], []
    for key, cls in [('cloud', PointCloud2), ('imu', Imu)]:
        spec = c['qos_topics'][key]
        pub = node.create_publisher(cls, spec['output'], outgoing)
        pubs.append(pub)
        counts[key] = dict(received=0, forwarded=0, duplicates=0, last_stamp_s=None)

        def receive(message, key=key, pub=pub):
            item = counts[key]
            item['received'] += 1
            stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            if key == 'imu':
                previous = item['last_stamp_s']
                if previous is not None and stamp <= previous:
                    item['duplicates'] += 1
                    item['last_stamp_s'] = previous
                    return
            # Publish original message object, without reconstructing fields.
            pub.publish(message)
            item['forwarded'] += 1
            item['last_stamp_s'] = stamp

        subs.append(node.create_subscription(cls, spec['input'], receive, incoming))
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = dict(motor_commands=0, messages_unchanged=True, counts=counts,
                  acceptance='QoS adaptation only; does not establish registration validity')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    try:
        while rclpy.ok() and not stopped and (a.seconds == 0 or time.monotonic()-start < a.seconds):
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        result.update(elapsed_s=time.monotonic()-start, stopped=True)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(result, indent=2))
    return 0 if all(v['forwarded'] for v in counts.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
