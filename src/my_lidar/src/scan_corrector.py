#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scan_corrector.py
#   - /scan 각도 보정 -> /scan_corrected (전체 스캔)
#   - ROI 안 포인트만  -> /scan_roi
#   - ROI 경계선       -> /roi_marker (RViz Marker)

import math
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

pub_scan      = None
pub_roi_scan  = None
pub_marker    = None


def make_roi_marker(x_min, x_max, y_min, y_max, frame_id):
    marker = Marker()
    marker.header.stamp    = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.ns              = "roi"
    marker.id              = 0
    marker.type            = Marker.LINE_STRIP
    marker.action          = Marker.ADD
    marker.scale.x         = 0.02       # 선 두께 (m)
    marker.color.r         = 0.0
    marker.color.g         = 1.0
    marker.color.b         = 0.0
    marker.color.a         = 1.0
    marker.lifetime        = rospy.Duration(0.2)

    # 직사각형 꼭짓점 (닫힌 루프)
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min),
    ]
    for x, y in corners:
        p = Point()
        p.x = x
        p.y = y
        p.z = 0.0
        marker.points.append(p)

    return marker


def scan_callback(msg):
    offset_rad = math.radians(rospy.get_param('~angle_offset_deg', 0.0))
    x_min = rospy.get_param('~roi_x_min', -0.5)
    x_max = rospy.get_param('~roi_x_max',  0.5)
    y_min = rospy.get_param('~roi_y_min',  0.1)
    y_max = rospy.get_param('~roi_y_max',  2.0)

    frame_id    = msg.header.frame_id
    angle_min   = msg.angle_min + offset_rad
    angle_max   = msg.angle_max + offset_rad
    angle_inc   = msg.angle_increment

    # /scan_corrected: 각도 보정된 전체 스캔
    corrected              = LaserScan()
    corrected.header       = msg.header
    corrected.angle_min    = angle_min
    corrected.angle_max    = angle_max
    corrected.angle_increment = angle_inc
    corrected.time_increment  = msg.time_increment
    corrected.scan_time    = msg.scan_time
    corrected.range_min    = msg.range_min
    corrected.range_max    = msg.range_max
    corrected.ranges       = msg.ranges
    corrected.intensities  = msg.intensities
    pub_scan.publish(corrected)

    # /scan_roi: ROI 안 포인트만 (밖은 0으로 마스킹)
    roi_ranges = list(msg.ranges)
    for i, r in enumerate(msg.ranges):
        if math.isnan(r) or math.isinf(r) or r < msg.range_min:
            roi_ranges[i] = 0.0
            continue
        rad = angle_min + i * angle_inc
        x = math.sin(rad) * r
        y = math.cos(rad) * r
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            roi_ranges[i] = 0.0

    roi_scan              = LaserScan()
    roi_scan.header       = msg.header
    roi_scan.angle_min    = angle_min
    roi_scan.angle_max    = angle_max
    roi_scan.angle_increment = angle_inc
    roi_scan.time_increment  = msg.time_increment
    roi_scan.scan_time    = msg.scan_time
    roi_scan.range_min    = msg.range_min
    roi_scan.range_max    = msg.range_max
    roi_scan.ranges       = roi_ranges
    roi_scan.intensities  = []
    pub_roi_scan.publish(roi_scan)

    # /roi_marker: ROI 직사각형 경계선
    pub_marker.publish(make_roi_marker(x_min, x_max, y_min, y_max, frame_id))


def main():
    global pub_scan, pub_roi_scan, pub_marker
    rospy.init_node('scan_corrector')
    pub_scan     = rospy.Publisher('/scan_corrected', LaserScan, queue_size=1)
    pub_roi_scan = rospy.Publisher('/scan_roi',       LaserScan, queue_size=1)
    pub_marker   = rospy.Publisher('/roi_marker', Marker, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()
