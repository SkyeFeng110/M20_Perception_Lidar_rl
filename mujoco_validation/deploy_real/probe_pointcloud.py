"""Read-only bounded ROS point-cloud probe. Does not publish any topic."""
import argparse
import json
import math
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topic', default='/LIDAR/POINTS')
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.seconds <= 120 or a.output.exists():
        p.error('bounded duration and a new output path required')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    rclpy.init(args=[])
    node = rclpy.create_node('m20_readonly_cloud_probe')
    result = dict(topic=a.topic, received=0, nonempty=0, samples=[], motor_commands=0)
    previous = None
    gaps, ages = [], []

    def receive(cloud):
        nonlocal previous
        now = time.monotonic()
        if previous is not None:
            gaps.append(now - previous)
        previous = now
        result['received'] += 1
        if cloud.width * cloud.height and cloud.data:
            result['nonempty'] += 1
        stamp = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
        age = time.time() - stamp
        if math.isfinite(age):
            ages.append(age)
        if len(result['samples']) < 3:
            result['samples'].append(dict(stamp_s=stamp, age_s=age, frame=cloud.header.frame_id,
                                         width=cloud.width, height=cloud.height, bytes=len(cloud.data),
                                         point_step=cloud.point_step,
                                         fields=[dict(name=f.name, offset=f.offset, datatype=f.datatype)
                                                 for f in cloud.fields]))

    subscription = node.create_subscription(PointCloud2, a.topic, receive, qos_profile_sensor_data)
    start = time.monotonic()
    try:
        while time.monotonic() - start < a.seconds:
            rclpy.spin_once(node, timeout_sec=.1)
        result.update(elapsed_s=time.monotonic()-start,
                      publishers=node.count_publishers(a.topic),
                      max_interarrival_s=max(gaps, default=None),
                      stamp_age_min_s=min(ages, default=None), stamp_age_max_s=max(ages, default=None))
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['nonempty'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
