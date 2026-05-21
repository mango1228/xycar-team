#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lidar_logger.py - /scan 토픽을 받아 장애물의 각도/거리를 로그로 출력
# 좌표계: angle=0 -> 정면(+y), angle>0 -> 오른쪽(+x), angle<0 -> 왼쪽(-x)
# 기준: xycar 라이다는 차량 전방에 장착, index 180 = 정면

import math
import rospy
from sensor_msgs.msg import LaserScan

DIST_MIN       = 0.05   # 유효 최소 거리 (m) - 센서 노이즈 제거
DIST_MAX       = 4.0    # 유효 최대 거리 (m)
CLUSTER_GAP    = 8      # 같은 장애물로 묶는 각도 간격 (도)
MIN_POINTS     = 3      # 클러스터 최소 포인트 수 (너무 작은 노이즈 제거)
FRONT_HALF_DEG = 180    # 전방 반원만 볼 경우 각도 제한 (180 = 전방 ±180 = 전체)
LOG_HZ         = 5      # 초당 최대 출력 횟수


def direction_str(angle_deg):
    if angle_deg < -30:
        return "왼쪽"
    elif angle_deg > 30:
        return "오른쪽"
    else:
        return "정면"


def cluster_obstacles(points):
    """(angle_deg, dist) 리스트 -> 클러스터별 (대표각도, 최근접거리, 방향) 리스트"""
    if not points:
        return []

    points = sorted(points, key=lambda p: p[0])  # 각도 순 정렬

    clusters = []
    cur = [points[0]]
    for angle, dist in points[1:]:
        if angle - cur[-1][0] <= CLUSTER_GAP:
            cur.append((angle, dist))
        else:
            clusters.append(cur)
            cur = [(angle, dist)]
    clusters.append(cur)

    result = []
    for cluster in clusters:
        if len(cluster) < MIN_POINTS:
            continue
        closest_angle, closest_dist = min(cluster, key=lambda p: p[1])
        result.append((closest_angle, closest_dist, direction_str(closest_angle)))

    return result


def scan_callback(data):
    angle_min = data.angle_min
    angle_inc = data.angle_increment
    ranges    = data.ranges

    points = []
    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r):
            continue
        if not (DIST_MIN < r < DIST_MAX):
            continue
        angle_deg = math.degrees(angle_min + i * angle_inc)
        if abs(angle_deg) > FRONT_HALF_DEG:
            continue
        points.append((angle_deg, r))

    obstacles = cluster_obstacles(points)

    if not obstacles:
        rospy.loginfo_throttle(2.0, "[lidar_logger] 감지된 장애물 없음")
        return

    # 거리 가까운 순으로 정렬해서 출력
    obstacles.sort(key=lambda o: o[1])

    parts = []
    for angle_deg, dist, direction in obstacles:
        parts.append("[%s] %+.0f도 %.2fm" % (direction, angle_deg, dist))

    rospy.loginfo_throttle(
        1.0 / LOG_HZ,
        "[lidar_logger] 장애물 %d개  %s" % (len(obstacles), "  |  ".join(parts))
    )


def main():
    rospy.init_node('lidar_logger')
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    rospy.loginfo("[lidar_logger] 시작 - /scan 구독 중...")
    rospy.spin()


if __name__ == '__main__':
    main()
