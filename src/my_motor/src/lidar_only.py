#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lidar_only.py - 라이다 전용 처리 노드
# 1단계: /scan 수신 + 통계 출력
# 2단계: ROI 초록 박스 RViz 표시
# 3단계: ROI 안 포인트 빨강 표시
# 4단계: 빈 공간 하늘색 원뿔 + 가장 큰 원뿔 중심 노란 화살표  ← 현재

import math
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as GeoPoint

# ROI 박스 (lane_drive.py와 동일 영역)
LIDAR_ROI_X     =  0.2   # 좌우 반폭 (±m)
LIDAR_ROI_Y_MIN = -0.6   # 전방 시작 (m, 음수=전방)
LIDAR_ROI_Y_MAX = -0.2   # 전방 끝 (m)

R_VIZ       = 0.7    # 빈 공간 원뿔 시각화 반경 (m)
MIN_GAP_ANG = 0.05   # 노이즈 무시 최소 갭 각도 (rad, ≈3°)

# ROI 각도 경계: near 모서리 기준 (laser_frame, x=cos*r, y=sin*r)
ROI_ANG_MIN = math.atan2(LIDAR_ROI_Y_MAX, -LIDAR_ROI_X)  # ≈ -135°
ROI_ANG_MAX = math.atan2(LIDAR_ROI_Y_MAX,  LIDAR_ROI_X)  # ≈  -45°

scan_data  = None
marker_pub = None


def get_roi_data(scan):
    """ROI 박스 안 포인트: [(x, y, angle)] 반환."""
    data = []
    for i, r in enumerate(scan.ranges):
        if math.isnan(r) or math.isinf(r) or r < 0.01:
            continue
        rad = scan.angle_min + i * scan.angle_increment
        x   = math.cos(rad) * r
        y   = math.sin(rad) * r
        if (-LIDAR_ROI_X <= x <= LIDAR_ROI_X and
                LIDAR_ROI_Y_MIN <= y <= LIDAR_ROI_Y_MAX):
            data.append((x, y, rad))
    return data


def get_gaps(data):
    """ROI 안 포인트 각도들 사이의 빈 구간 [(a1, a2)] 반환.
    포인트 없으면 ROI 전체가 하나의 빈 공간. MIN_GAP_ANG 미만은 노이즈로 무시."""
    if not data:
        return [(ROI_ANG_MIN, ROI_ANG_MAX)]

    angles = sorted(d[2] for d in data)
    angles = [a for a in angles if ROI_ANG_MIN <= a <= ROI_ANG_MAX]
    if not angles:
        return [(ROI_ANG_MIN, ROI_ANG_MAX)]

    raw = ([(ROI_ANG_MIN, angles[0])] +
           [(angles[i], angles[i + 1]) for i in range(len(angles) - 1)] +
           [(angles[-1], ROI_ANG_MAX)])
    return [(a1, a2) for a1, a2 in raw if a2 - a1 >= MIN_GAP_ANG]


def publish_roi_markers(scan):
    arr      = MarkerArray()
    stamp    = scan.header.stamp
    frame_id = scan.header.frame_id

    data = get_roi_data(scan)
    gaps = get_gaps(data)

    def pt(x, y, z=0.0):
        p = GeoPoint(); p.x = x; p.y = y; p.z = z; return p

    # ── 1. ROI 박스 (초록 LINE_STRIP) ──────────────────────────────
    box = Marker()
    box.header.stamp = stamp; box.header.frame_id = frame_id
    box.ns = "roi"; box.id = 0; box.type = Marker.LINE_STRIP
    box.action = Marker.ADD
    box.pose.orientation.w = 1.0
    box.scale.x = 0.02
    box.color.r = 0.0; box.color.g = 1.0; box.color.b = 0.0; box.color.a = 1.0
    box.lifetime = rospy.Duration(0.3)
    for cx, cy in [(-LIDAR_ROI_X, LIDAR_ROI_Y_MIN),
                   ( LIDAR_ROI_X, LIDAR_ROI_Y_MIN),
                   ( LIDAR_ROI_X, LIDAR_ROI_Y_MAX),
                   (-LIDAR_ROI_X, LIDAR_ROI_Y_MAX),
                   (-LIDAR_ROI_X, LIDAR_ROI_Y_MIN)]:
        box.points.append(pt(cx, cy))
    arr.markers.append(box)

    # ── 2. ROI 안 장애물 포인트 (빨강 POINTS) ───────────────────────
    pm = Marker()
    pm.header.stamp = stamp; pm.header.frame_id = frame_id
    pm.ns = "roi"; pm.id = 1; pm.type = Marker.POINTS
    pm.action = Marker.ADD if data else Marker.DELETE
    pm.pose.orientation.w = 1.0
    pm.scale.x = 0.05; pm.scale.y = 0.05
    pm.color.r = 1.0; pm.color.g = 0.0; pm.color.b = 0.0; pm.color.a = 1.0
    pm.lifetime = rospy.Duration(0.3)
    for x, y, _ in data:
        pm.points.append(pt(x, y))
    arr.markers.append(pm)

    # ── 3. 빈 공간 원뿔 (하늘색 TRIANGLE_LIST) ──────────────────────
    cones = Marker()
    cones.header.stamp = stamp; cones.header.frame_id = frame_id
    cones.ns = "roi"; cones.id = 2; cones.type = Marker.TRIANGLE_LIST
    cones.action = Marker.ADD if gaps else Marker.DELETE
    cones.pose.orientation.w = 1.0
    cones.scale.x = 1.0; cones.scale.y = 1.0; cones.scale.z = 1.0
    cones.color.r = 0.0; cones.color.g = 0.8; cones.color.b = 1.0; cones.color.a = 0.35
    cones.lifetime = rospy.Duration(0.3)
    origin = pt(0.0, 0.0)
    for a1, a2 in gaps:
        n = max(2, int((a2 - a1) / 0.05))
        for k in range(n):
            ang0 = a1 + (a2 - a1) * k / n
            ang1 = a1 + (a2 - a1) * (k + 1) / n
            p0 = pt(math.cos(ang0) * R_VIZ, math.sin(ang0) * R_VIZ)
            p1 = pt(math.cos(ang1) * R_VIZ, math.sin(ang1) * R_VIZ)
            cones.points.extend([origin, p0, p1])
    arr.markers.append(cones)

    # ── 4. 가장 큰 빈 공간 중심 (노란 ARROW) ────────────────────────
    arrow = Marker()
    arrow.header.stamp = stamp; arrow.header.frame_id = frame_id
    arrow.ns = "roi"; arrow.id = 3; arrow.type = Marker.ARROW
    arrow.pose.orientation.w = 1.0
    arrow.scale.x = 0.03; arrow.scale.y = 0.07; arrow.scale.z = 0.07
    arrow.color.r = 1.0; arrow.color.g = 1.0; arrow.color.b = 0.0; arrow.color.a = 1.0
    arrow.lifetime = rospy.Duration(0.3)
    if gaps:
        largest  = max(gaps, key=lambda g: g[1] - g[0])
        bisector = (largest[0] + largest[1]) / 2.0
        arrow.action = Marker.ADD
        arrow.points = [origin,
                        pt(math.cos(bisector) * 0.5, math.sin(bisector) * 0.5)]
    else:
        arrow.action = Marker.DELETE
    arr.markers.append(arrow)

    marker_pub.publish(arr)


def scan_callback(data):
    global scan_data
    scan_data = data
    publish_roi_markers(data)


def main():
    global marker_pub
    rospy.init_node('lidar_only')
    marker_pub = rospy.Publisher('/lidar_only/roi_markers', MarkerArray, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    print("lidar_only started - /scan 수신 대기 중")

    rate  = rospy.Rate(2)
    count = 0
    while not rospy.is_shutdown():
        if scan_data is None:
            rate.sleep()
            continue
        n     = len(scan_data.ranges)
        valid = [r for r in scan_data.ranges
                 if not (math.isnan(r) or math.isinf(r)) and r > 0.01]
        count += 1
        if valid:
            print("[%d] N=%d valid=%d r_min=%.2f r_max=%.2f"
                  % (count, n, len(valid), min(valid), max(valid)))
        else:
            print("[%d] N=%d 유효 거리 없음" % (count, n))
        rate.sleep()


if __name__ == '__main__':
    main()
