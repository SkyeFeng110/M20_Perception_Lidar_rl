"""Read-only live check: both QoS bridge outputs must preserve serialized inputs."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.seconds <= 30 or a.output.exists():
        p.error('Bounded probe and a new output path required')
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Imu, PointCloud2
    c = json.loads((Path(__file__).resolve().parent/'lidar_source_contract.json').read_text())
    rclpy.init(args=[])
    node = rclpy.create_node('m20_readonly_qos_integrity_probe')
    pending = {key: {'input': OrderedDict(), 'output': OrderedDict()} for key in c['qos_topics']}
    result = {key: dict(matched=0, mismatched=0, serialization_variations=0, variation_positions=[]) for key in pending}
    subscriptions = []

    def receive(message, key, side):
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        serialized = serialize_message(message)
        values = dict(stamp=stamp, frame=message.header.frame_id)
        if hasattr(message, 'data'):
            values.update(width=message.width, height=message.height, point_step=message.point_step,
                          row_step=message.row_step, is_bigendian=message.is_bigendian,
                          is_dense=message.is_dense,
                          fields=[(f.name,f.offset,f.datatype,f.count) for f in message.fields],
                          data_sha256=hashlib.sha256(bytes(message.data)).hexdigest())
        else:
            values.update(orientation=[message.orientation.x,message.orientation.y,message.orientation.z,message.orientation.w],
                          angular_velocity=[message.angular_velocity.x,message.angular_velocity.y,message.angular_velocity.z],
                          linear_acceleration=[message.linear_acceleration.x,message.linear_acceleration.y,message.linear_acceleration.z],
                          orientation_covariance=list(message.orientation_covariance),
                          angular_velocity_covariance=list(message.angular_velocity_covariance),
                          linear_acceleration_covariance=list(message.linear_acceleration_covariance))
        canonical = hashlib.sha256(json.dumps(values,sort_keys=True,allow_nan=False).encode()).hexdigest()
        digest = (serialized, canonical)
        other = 'output' if side == 'input' else 'input'
        match = pending[key][other].pop(stamp, None)
        if match is not None:
            result[key]['matched' if match[1] == digest[1] else 'mismatched'] += 1
            if match[1] == digest[1] and match[0] != digest[0]:
                result[key]['serialization_variations'] += 1
                if not result[key]['variation_positions']:
                    result[key]['variation_positions'] = [i for i,(x,y) in enumerate(zip(match[0],digest[0])) if x!=y][:32]
        else:
            pending[key][side][stamp] = digest
            while len(pending[key][side]) > 256:
                pending[key][side].popitem(last=False)

    for key, cls in [('cloud', PointCloud2), ('imu', Imu)]:
        for side in ('input', 'output'):
            qos = QoSProfile(depth=10, reliability=(ReliabilityPolicy.BEST_EFFORT if side == 'input'
                                                  else ReliabilityPolicy.RELIABLE))
            subscriptions.append(node.create_subscription(cls, c['qos_topics'][key][side],
                                 lambda message, key=key, side=side: receive(message, key, side), qos))
    start = time.monotonic()
    try:
        while time.monotonic()-start < a.seconds:
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    passed = all(v['matched'] > 0 and v['mismatched'] == 0 for v in result.values())
    report = dict(motor_commands=0, elapsed_s=time.monotonic()-start, channels=result, pass_=passed,
                  comparison='All public fields and exact raw data SHA256 paired by original timestamp; CDR byte variations reported separately')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
