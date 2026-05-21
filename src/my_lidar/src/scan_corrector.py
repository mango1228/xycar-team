#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scan_corrector.py - /scan 각도 오프셋 보정 후 /scan_corrected 재발행
# angle_min/angle_max 에 오프셋을 더하는 것만으로 전체 스캔이 회전됨

import math
import rospy
from sensor_msgs.msg import LaserScan

pub = None

def scan_callback(msg):
    offset_rad = math.radians(rospy.get_param('~angle_offset_deg', 0.0))

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
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    main()
