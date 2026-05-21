#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scan_corrector.py - /scan 각도 오프셋 보정 후 /scan_corrected 재발행
# rqt_reconfigure 슬라이더로 angle_offset_deg 실시간 조정 가능

import math
import rospy
from sensor_msgs.msg import LaserScan
from dynamic_reconfigure.server import Server
from my_lidar.cfg import ScanCorrectorConfig

offset_rad = 0.0
pub = None

def reconfigure_callback(config, level):
    global offset_rad
    offset_rad = math.radians(config.angle_offset_deg)
    rospy.loginfo("[scan_corrector] angle_offset_deg = %.1f" % config.angle_offset_deg)
    return config

def scan_callback(msg):
    corrected = LaserScan()
    corrected.header          = msg.header
    corrected.angle_min       = msg.angle_min       + offset_rad
    corrected.angle_max       = msg.angle_max       + offset_rad
    corrected.angle_increment = msg.angle_increment
    corrected.time_increment  = msg.time_increment
    corrected.scan_time       = msg.scan_time
    corrected.range_min       = msg.range_min
    corrected.range_max       = msg.range_max
    corrected.ranges          = msg.ranges
    corrected.intensities     = msg.intensities
    pub.publish(corrected)

def main():
    global pub
    rospy.init_node('scan_corrector')
    pub = rospy.Publisher('/scan_corrected', LaserScan, queue_size=1)
    Server(ScanCorrectorConfig, reconfigure_callback)
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    main()
