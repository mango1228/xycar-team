#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lidar_only.py - 라이다 전용 처리 노드
# 1단계: /scan 수신 + 통계 출력
# 2단계: ROI(직사각형) 초록 박스 RViz 표시  ← 현재
# 다음 단계에서 ROI 안 점 강조 / 제어 로직 추가 예정.

import math
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as GeoPoint

# ROI 박스 (lane_drive.py와 동일 영역)
LIDAR_ROI_X     =  0.2   # 좌우 반폭 (±m)
LIDAR_ROI_Y_MIN = -0.6   # 전방 시작 (m, 음수=전방)
LIDAR_ROI_Y_MAX = -0.2   # 전방 끝 (m)

scan_data  = None
marker_pub = None


def publish_roi_box(scan):
    """ROI 박스(초록 LINE_STRIP) 마커 발행"""
    arr = MarkerArray()
    box = Marker()
    box.header.stamp    = scan.header.stamp
    box.header.frame_id = scan.header.frame_id
    box.ns   = "roi"
    box.id   = 0
    box.type = Marker.LINE_STRIP
    box.action = Marker.ADD
    box.scale.x = 0.02
    box.color.r = 0.0; box.color.g = 1.0; box.color.b = 0.0; box.color.a = 1.0
    box.lifetime = rospy.Duration(0.3)
    for cx, cy in [(-LIDAR_ROI_X, LIDAR_ROI_Y_MIN),
                   ( LIDAR_ROI_X, LIDAR_ROI_Y_MIN),
                   ( LIDAR_ROI_X, LIDAR_ROI_Y_MAX),
                   (-LIDAR_ROI_X, LIDAR_ROI_Y_MAX),
                   (-LIDAR_ROI_X, LIDAR_ROI_Y_MIN)]:
        p = GeoPoint(); p.x = cx; p.y = cy; p.z = 0.0
        box.points.append(p)
    arr.markers.append(box)
    marker_pub.publish(arr)


def scan_callback(data):
    global scan_data
    scan_data = data
    publish_roi_box(data)


def main():
    global marker_pub
    rospy.init_node('lidar_only')
    marker_pub = rospy.Publisher('/lidar_only/roi_markers', MarkerArray, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    print("lidar_only started - /scan 수신 대기 중")

    rate = rospy.Rate(2)   # 2 Hz 로 통계 출력
    count = 0
    while not rospy.is_shutdown():
        if scan_data is None:
            rate.sleep()
            continue

        n = len(scan_data.ranges)
        valid = [r for r in scan_data.ranges
                 if not (math.isnan(r) or math.isinf(r)) and r > 0.01]
        count += 1
        if valid:
            print("[%d] N=%d valid=%d r_min=%.2f r_max=%.2f angle=[%.2f, %.2f] rad"
                  % (count, n, len(valid), min(valid), max(valid),
                     scan_data.angle_min, scan_data.angle_max))
        else:
            print("[%d] N=%d 유효 거리 없음" % (count, n))

        rate.sleep()


if __name__ == '__main__':
    main()
