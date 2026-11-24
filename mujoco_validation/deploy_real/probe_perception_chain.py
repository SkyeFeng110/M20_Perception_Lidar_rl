"""Bounded, passive ROS sensor/LIO/registration/6D timing probe; never publishes."""
import argparse
import collections
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',type=int,choices=(0,87),required=True)
    parser.add_argument('--seconds',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.seconds<=120 or args.output.exists():
        parser.error('Bounded duration and new output path required')
    os.environ['ROS_DOMAIN_ID']=str(args.domain)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2,Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    topics=([(PointCloud2,'/LIDAR/POINTS'),(PointCloud2,'/m20_deploy/lio_input/points'),
             (Imu,'/IMU'),(Imu,'/m20_deploy/lio_input/imu'),
             (PointCloud2,'/staircase_gos_direct/lio/cloud_world'),
             (Odometry,'/staircase_gos_direct/lio/odom'),(String,'/M20_V11_STAIRCASE_PACKET')]
            if args.domain==0 else [(PointCloud2,'/staircase_gos_direct/registered_point_cloud'),
                                    (Odometry,'/staircase_gos_direct/odometry'),
                                    (String,'/staircase_gos_direct/source_status')])
    rclpy.init(args=[]);node=rclpy.create_node('m20_passive_chain_probe')
    results={};subscriptions=[]
    def callback(topic,message):
        row=results[topic];now=time.monotonic();wall=time.time()
        row['received']+=1
        if row['last_receipt_mono_s'] is not None:
            row['max_gap_s']=max(row['max_gap_s'],now-row['last_receipt_mono_s'])
        row['last_receipt_mono_s']=now
        if hasattr(message,'header'):
            stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            previous=row['last_stamp_s']
            if previous is not None and stamp<=previous:row['nonincreasing_stamps']+=1
            if previous is not None:row['max_stamp_gap_s']=max(row['max_stamp_gap_s'],stamp-previous)
            row['last_stamp_s']=stamp
            age=wall-stamp
            row['age_min_s']=age if row['age_min_s'] is None else min(row['age_min_s'],age)
            row['age_max_s']=age if row['age_max_s'] is None else max(row['age_max_s'],age)
            if isinstance(message,PointCloud2):row['last_points']=message.width*message.height
            if isinstance(message,Odometry):
                p=message.pose.pose.position;row['last_position']=[p.x,p.y,p.z]
        else:
            try:value=json.loads(message.data)
            except (TypeError,json.JSONDecodeError):row['malformed_json']+=1;return
            row['last_payload']=value
            if isinstance(value,dict):
                current=value.get('current',value)
                if isinstance(current,dict):row['reasons'][str(current.get('reason','unspecified'))]+=1
    for cls,topic in topics:
        results[topic]=dict(received=0,max_gap_s=0.,last_receipt_mono_s=None,last_stamp_s=None,
                            nonincreasing_stamps=0,max_stamp_gap_s=0.,age_min_s=None,age_max_s=None,
                            reasons=collections.Counter(),malformed_json=0)
        # Foxy invokes subscription callbacks with only the message.
        def make_receive(name):
            def receive(message):callback(name,message)
            return receive
        subscriptions.append(node.create_subscription(cls,topic,make_receive(topic),qos_profile_sensor_data))
    started=time.monotonic()
    try:
        while time.monotonic()-started<args.seconds:rclpy.spin_once(node,timeout_sec=.05)
        for topic,row in results.items():
            row['publishers']=node.count_publishers(topic)
            row['publisher_endpoints']=[dict(node_name=p.node_name,node_namespace=p.node_namespace,
                                            gid=bytes(p.endpoint_gid).hex())
                                       for p in node.get_publishers_info_by_topic(topic)]
            row['last_receipt_age_s']=None if row['last_receipt_mono_s'] is None else time.monotonic()-row['last_receipt_mono_s']
        report=dict(domain=args.domain,elapsed_s=time.monotonic()-started,motor_commands=0,
                    created_publishers=0,topics=results)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
