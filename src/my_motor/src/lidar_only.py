#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lidar_only.py - 라이다 전용 처리 노드
# 1단계: /scan 수신 확인 + 통계 출력 (RViz로 시각화만 확인)
# 다음 단계에서 ROI/제어 로직 추가 예정.

import math
import rospy
from sensor_msgs.msg import LaserScan

scan_data = None


def scan_callback(data):
    global scan_data
    scan_data = data


def main():
    rospy.init_node('lidar_only')
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
