#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lane_drive.py - 허프변환 기반 차선주행 노드
# 베이스: auto_drive/hough_drive.py 구조 (ROI 먼저 자르고 Hough)
# 개선: 튜닝 Canny / 조향 클램프+게인 / 깔끔한 종료
#       한쪽 차선만 보일 때 -> 보이는 차선에서 중점을 직접 재구성 (반폭 EMA 사용)
#       라이다: lidar_only와 동일 방식 (laser_frame, cos/sin 표준 좌표)

import os
import time
import rospy
import numpy as np
import cv2, math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as GeoPoint
from xycar_msgs.msg import xycar_motor

# ===== 튜닝 파라미터 =====
# LiDAR ROI 박스 (laser_frame 기준, x=cos*r, y=sin*r)
LIDAR_ROI_X     =  0.33   # 좌우 반폭 (±m)
LIDAR_ROI_Y_MIN = -0.55   # 전방 시작 (m)
LIDAR_ROI_Y_MAX = -0.05   # 전방 끝 (m)

R_VIZ           = 0.7    # 빈 공간 원뿔 시각화 반경 (m)
MIN_GAP_ANG     = 0.05   # 노이즈 무시 최소 갭 각도 (rad, ≈3°)
LIDAR_CENTER_GAIN = -130.0  # 라이다 bisector → 픽셀 오프셋 게인

# ROI 각도 경계 (near 모서리 기준)
ROI_ANG_MIN    = math.atan2(LIDAR_ROI_Y_MAX, -LIDAR_ROI_X)  # ≈ -135°
ROI_ANG_MAX    = math.atan2(LIDAR_ROI_Y_MAX,  LIDAR_ROI_X)  # ≈  -45°
ROI_ANG_CENTER = (ROI_ANG_MIN + ROI_ANG_MAX) / 2.0           # ≈  -90° (차량 전방)

CANNY_LOW  = 40
CANNY_HIGH = 100
OFFSET     = 330     # 카메라 ROI 띠 시작 row
GAP        = 110      # 카메라 ROI 띠 높이
GAIN              = 0.25    # P 게인
GAIN_I            = 0.01    # I 게인 (0이면 비활성)
GAIN_D            = 0.02    # D 게인 (0이면 비활성)
SPEED             = 5
EMA_ALPHA         = 0.3
ONE_LANE_RATIO    = 0.7    # 한쪽 차선만 보일 때 추종 거리 비율 (1.0=원래 반폭, <1=차선에 더 가깝게)
SHOW_DEBUG = True

if SHOW_DEBUG and not os.environ.get('DISPLAY'):
    print("[lane_drive] DISPLAY 없음 - 디버그 창 비활성화 (창 보려면 ssh -Y 로 접속)")
    SHOW_DEBUG = False

WIDTH, HEIGHT = 640, 480

HOUGH_THRESHOLD = 30
HOUGH_MIN_LEN   = 20
HOUGH_MAX_GAP   = 10
CENTER_MARGIN   = 90

image      = np.empty(shape=[0])
lidar_scan = None
bridge     = CvBridge()
motor_pub  = None
marker_pub = None
motor_msg  = xycar_motor()

ema_half_width = None
prev_center    = WIDTH // 2

# PID 상태
prev_error = 0.0
i_error    = 0.0
pid_time   = None


def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")


def lidar_callback(data):
    global lidar_scan
    lidar_scan = data


def get_roi_data(scan):
    """LaserScan -> ROI 박스 안 포인트 [(x, y, angle)] 반환.
    laser_frame 표준 좌표 (x=cos*r, y=sin*r). scan이 None이면 []."""
    if scan is None:
        return []
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
    """ROI 안 포인트 각도 사이 빈 구간 [(a1, a2)] 반환.
    포인트 없으면 ROI 전체가 하나의 빈 공간. MIN_GAP_ANG 미만은 무시."""
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


def publish_roi_markers(data, gaps, bisector, scan):
    """RViz 마커 발행: 초록 박스 / 빨강 포인트 / 하늘색 원뿔 / 노란 화살표"""
    if scan is None:
        return
    stamp    = scan.header.stamp
    frame_id = scan.header.frame_id
    arr      = MarkerArray()

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
    box.lifetime = rospy.Duration(0.1)
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
    pm.lifetime = rospy.Duration(0.1)
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
    cones.lifetime = rospy.Duration(0.1)
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
    arrow.lifetime = rospy.Duration(0.1)
    if bisector is not None:
        arrow.action = Marker.ADD
        arrow.points = [origin,
                        pt(math.cos(bisector) * 0.5, math.sin(bisector) * 0.5)]
    else:
        arrow.action = Marker.DELETE
    arr.markers.append(arrow)

    marker_pub.publish(arr)


def drive(angle, speed):
    motor_msg.angle = int(angle)
    motor_msg.speed = int(speed)
    motor_pub.publish(motor_msg)


def divide_left_right(lines):
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = float(y2 - y1) / float(x2 - x1)
        if abs(slope) < 0.1 or abs(slope) > 10:
            continue
        if slope < 0 and x2 < WIDTH/2 - CENTER_MARGIN:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > WIDTH/2 + CENTER_MARGIN:
            right.append((x1, y1, x2, y2))
    return left, right


def get_pos(lines):
    if len(lines) == 0:
        return None
    x_sum = y_sum = m_sum = 0.0
    for x1, y1, x2, y2 in lines:
        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)
    n = len(lines)
    x_avg = x_sum / (n * 2)
    y_avg = y_sum / (n * 2)
    m = m_sum / n
    if m == 0:
        return None
    b = y_avg - m * x_avg
    return int((GAP/2 - b) / m)


def process(frame):
    """프레임 -> (center, mode, 좌/우 선분 리스트, lpos, rpos)"""
    global ema_half_width, prev_center

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    roi   = edge[OFFSET:OFFSET+GAP, 0:WIDTH]
    lines = cv2.HoughLinesP(roi, 1, math.pi/180, HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)

    if lines is None:
        return prev_center, "NONE", [], [], None, None

    left, right = divide_left_right(lines)
    lpos = get_pos(left)
    rpos = get_pos(right)

    if lpos is not None and rpos is not None:
        center = (lpos + rpos) // 2
        half = (rpos - lpos) / 2.0
        if ema_half_width is None:
            ema_half_width = half
        else:
            ema_half_width = EMA_ALPHA * half + (1 - EMA_ALPHA) * ema_half_width
        mode = "BOTH"
    elif lpos is not None:
        # 왼쪽만 보임 -> 기억한 반폭의 ONE_LANE_RATIO 만큼 오른쪽으로
        center = int(lpos + ema_half_width * ONE_LANE_RATIO) if ema_half_width is not None else prev_center
        mode = "LEFT"
    elif rpos is not None:
        # 오른쪽만 보임 -> 기억한 반폭의 ONE_LANE_RATIO 만큼 왼쪽으로
        center = int(rpos - ema_half_width * ONE_LANE_RATIO) if ema_half_width is not None else prev_center
        mode = "RIGHT"
    else:
        center = prev_center
        mode = "NONE"

    prev_center = center
    return center, mode, left, right, lpos, rpos


def draw_debug(frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos):
    y = OFFSET + GAP // 2
    cv2.rectangle(frame, (0, OFFSET), (WIDTH-1, OFFSET+GAP), (0, 255, 0), 2)
    for x1, y1, x2, y2 in left:
        cv2.line(frame, (x1, y1+OFFSET), (x2, y2+OFFSET), (0, 0, 255), 2)
    for x1, y1, x2, y2 in right:
        cv2.line(frame, (x1, y1+OFFSET), (x2, y2+OFFSET), (255, 0, 0), 2)
    # 차선 검출 위치 (get_pos 반환값) 검정 점
    if lpos is not None:
        cv2.circle(frame, (lpos, y), 6, (0, 0, 0), -1)
    if rpos is not None:
        cv2.circle(frame, (rpos, y), 6, (0, 0, 0), -1)
    cv2.circle(frame, (cam_center, y), 6, (255, 128, 0),   -1)
    if lidar_c is not None:
        cv2.circle(frame, (lidar_c, y), 6, (0, 0, 255),    -1)
    cv2.circle(frame, (center, y),     8, (0, 255, 255),    2)
    cv2.circle(frame, (WIDTH//2, y),   6, (255, 255, 255), -1)
    cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('lane_drive', frame)
    cv2.waitKey(1)


def main():
    global motor_pub, marker_pub
    rospy.init_node('lane_drive')
    motor_pub  = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    marker_pub = rospy.Publisher('/lane_drive/roi_markers', MarkerArray, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    rospy.Subscriber('/scan', LaserScan, lidar_callback, queue_size=1)
    rospy.on_shutdown(lambda: drive(0, 0))

    rospy.sleep(2.0)
    print("lane_drive started")

    count = 0
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if image.size == 0:
            rate.sleep()
            continue

        frame = image.copy()
        center, mode, left, right, lpos, rpos = process(frame)
        cam_center = center

        roi_data = get_roi_data(lidar_scan)
        gaps     = get_gaps(roi_data)

        bisector = None
        lidar_c  = None
        if gaps:
            largest  = max(gaps, key=lambda g: g[1] - g[0])
            bisector = (largest[0] + largest[1]) / 2.0
            # ROI_ANG_CENTER - bisector: laser_frame에서 car left/right 방향 보정
            # bisector > CENTER(=-90°) → car's left → negative offset → steer left
            dev = ROI_ANG_CENTER - bisector
            lidar_c = max(0, min(WIDTH - 1, int(WIDTH // 2 + dev * LIDAR_CENTER_GAIN)))

        publish_roi_markers(roi_data, gaps, bisector, lidar_scan)

        if lidar_c is not None:
            # 차선 경계 안으로 클램프 (차선 바깥 조향 방지)
            if lpos is not None:
                lidar_c = max(lidar_c, lpos)
            if rpos is not None:
                lidar_c = min(lidar_c, rpos)
            if abs(lidar_c - WIDTH // 2) > abs(center - WIDTH // 2):
                center = lidar_c
                mode = mode + "+LIDAR"

        # PID 제어
        global prev_error, i_error, pid_time
        now = time.time()
        dt  = (now - pid_time) if pid_time is not None else 1e-6
        dt  = max(dt, 1e-6)
        pid_time = now

        error   = center - WIDTH // 2
        i_error += error * dt
        d_out   = (error - prev_error) / dt
        prev_error = error

        angle = error * GAIN + i_error * GAIN_I + d_out * GAIN_D
        angle = max(-50, min(50, angle))

        drive(angle, SPEED)

        if SHOW_DEBUG:
            draw_debug(frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos)

        count += 1
        if count % 30 == 0:
            half_str = "%.0f" % ema_half_width if ema_half_width is not None else "-"
            print("mode=%s center=%d angle=%d ema_half=%s"
                  % (mode, center, int(angle), half_str))

        rate.sleep()


if __name__ == '__main__':
    main()
