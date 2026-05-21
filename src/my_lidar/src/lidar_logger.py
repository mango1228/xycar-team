#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lidar_logger.py - /scan 토픽을 받아 장애물의 XY 좌표를 로그로 출력
# 좌표계: x=sin(rad)*dist (오른쪽 +), y=cos(rad)*dist (전방 +)
# xycar 라이다는 차량 전방에 장착 (xycar.md 15.3절 좌표계 기준)

import math
import rospy
from sensor_msgs.msg import LaserScan

DIST_MIN      = 0.05  # 유효 최소 거리 (m)
DIST_MAX      = 4.0   # 유효 최대 거리 (m)
CLUSTER_GAP_M = 0.15  # 같은 장애물로 묶는 XY 거리 간격 (m)
MIN_POINTS    = 3     # 클러스터 최소 포인트 수
LOG_HZ        = 5     # 초당 최대 출력 횟수


def xy_from_scan(ranges, angle_min, angle_inc):
    """LaserScan ranges -> 유효한 (x, y) 리스트"""
    pts = []
    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r):
            continue
        if not (DIST_MIN < r < DIST_MAX):
            continue
        rad = angle_min + i * angle_inc
        x = math.sin(rad) * r   # 오른쪽 +
        y = math.cos(rad) * r   # 전방 +
        pts.append((x, y))
    return pts


def cluster_xy(pts):
    """(x,y) 리스트를 거리 기반으로 클러스터링 -> 클러스터 리스트"""
    if not pts:
        return []

    pts = sorted(pts, key=lambda p: math.atan2(p[0], p[1]))  # 각도 순 정렬

    clusters = [[pts[0]]]
    for x, y in pts[1:]:
        cx, cy = clusters[-1][-1]
        if math.sqrt((x - cx)**2 + (y - cy)**2) <= CLUSTER_GAP_M:
            clusters[-1].append((x, y))
        else:
            clusters.append([(x, y)])

    return clusters


def obstacle_repr(cluster):
    """클러스터 -> 중심 XY (가장 가까운 포인트 기준)"""
    closest = min(cluster, key=lambda p: math.sqrt(p[0]**2 + p[1]**2))
    x, y = closest
    dist = math.sqrt(x**2 + y**2)

    if y < 0:
        fore = "후방"
    elif abs(x) < 0.1:
        fore = "정면"
    elif x > 0:
        fore = "오른쪽"
    else:
        fore = "왼쪽"

    return x, y, dist, fore


def scan_callback(data):
    pts = xy_from_scan(data.ranges, data.angle_min, data.angle_increment)
    clusters = cluster_xy(pts)
    obstacles = [c for c in clusters if len(c) >= MIN_POINTS]

    if not obstacles:
        rospy.loginfo_throttle(2.0, "[lidar] 감지된 장애물 없음")
        return

    infos = [obstacle_repr(c) for c in obstacles]
    infos.sort(key=lambda o: o[2])  # 가까운 순

    parts = []
    for x, y, dist, fore in infos:
        parts.append("[%s] x=%+.2fm y=%.2fm (%.2fm)" % (fore, x, y, dist))

    rospy.loginfo_throttle(
        1.0 / LOG_HZ,
        "[lidar] 장애물 %d개  %s" % (len(infos), "  |  ".join(parts))
    )


def main():
    rospy.init_node('lidar_logger')
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    rospy.loginfo("[lidar] 시작 (전방+y 오른쪽+x)")
    rospy.spin()


if __name__ == '__main__':
    main()
