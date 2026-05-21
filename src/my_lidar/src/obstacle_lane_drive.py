#!/usr/bin/env python
# -*- coding: utf-8 -*-

# obstacle_lane_drive.py
# 허프변환 차선 주행 + LiDAR 장애물 회피 통합 노드
#
# [지원 주행 모드]
#   MODE 1 (LANE_PARTIAL) : 한쪽 차선만 보임 – EMA 반폭으로 중점 추정
#   MODE 2 (AVOID)        : 차선 경계/내부 장애물 – LiDAR 회피 오프셋 적용
#   MODE 3 (TUNNEL)       : 양쪽 차선 모두 가림 – LiDAR 좌우 벽 균등 추종
#
# [주행 우선순위]  AVOID > TUNNEL > LANE_PARTIAL > LANE_BOTH > LANE_NONE

import os
import math
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import xycar_motor

# ====================================================================
# 튜닝 파라미터
# ====================================================================

# -- 카메라 / 차선 --
CANNY_LOW   = 40
CANNY_HIGH  = 100
OFFSET      = 340       # ROI 시작 행
GAP         = 40        # ROI 높이
GAIN        = 0.4       # 차선 P 게인
SPEED       = 3         # 주행 속도 (0~5)
EMA_ALPHA   = 0.3       # 차선 반폭 EMA 계수
CENTER_MARGIN = 90      # 좌/우 분리 여유 픽셀
WIDTH, HEIGHT = 640, 480

HOUGH_THRESHOLD = 30
HOUGH_MIN_LEN   = 20
HOUGH_MAX_GAP   = 10

# -- LiDAR --
# 장애물 감지 ROI (XY 좌표, 단위: m)
ANGLE_OFFSET_DEG = -4.0  # 라이다 장착 각도 보정
ROI_X            = 0.2   # 좌우 범위 (-ROI_X ~ +ROI_X)
ROI_X_MIN        = -ROI_X
ROI_X_MAX        =  ROI_X
ROI_Y_MIN        = -0.6  # 전후 시작
ROI_Y_MAX        = -0.1  # 전후 끝
MIN_ROI_POINTS   = 2     # ROI 안 포인트가 이 수 이상이면 장애물로 판정

# 터널 벽 감지 (측면, 각도 인덱스 기반 유지)
SIDE_HALF_DEG   = 60
LIDAR_TOTAL     = 360
LEFT_IDX_S      = LIDAR_TOTAL - SIDE_HALF_DEG
RIGHT_IDX_E     = SIDE_HALF_DEG

# 장애물 판정 거리 임계값 (미터)
OBSTACLE_DIST    = 0.7
TUNNEL_DIST      = 0.9

# 터널 모드: LiDAR 좌우 거리 균등 제어 게인
TUNNEL_GAIN      = 30.0  # (좌거리 - 우거리) * TUNNEL_GAIN → 조향각

# 회피 오프셋: 장애물 위치에 따라 중점을 밀어내는 픽셀 양
AVOID_OFFSET_PX  = 80    # 픽셀 단위 중점 이동량

# -- 표시 --
SHOW_DEBUG = True
if SHOW_DEBUG and not os.environ.get('DISPLAY'):
    print("[obstacle_lane_drive] DISPLAY 없음 - 디버그 창 비활성화")
    SHOW_DEBUG = False

# ====================================================================
# 전역 상태
# ====================================================================

image      = np.empty(shape=[0])
lidar_scan = None
bridge       = CvBridge()
motor_pub    = None
motor_msg    = xycar_motor()

ema_half_width = None
prev_center    = WIDTH // 2

# 현재 주행 상태 (디버그 출력용)
drive_mode = "INIT"


# ====================================================================
# 콜백
# ====================================================================

def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")


def lidar_callback(data):
    global lidar_scan
    lidar_scan = data


def drive(angle, speed):
    motor_msg.angle = int(np.clip(angle, -50, 50))
    motor_msg.speed = int(speed)
    motor_pub.publish(motor_msg)


# ====================================================================
# 차선 검출 (기존 lane_drive.py 구조 유지)
# ====================================================================

def divide_left_right(lines):
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = float(y2 - y1) / float(x2 - x1)
        if abs(slope) < 0.1 or abs(slope) > 10:
            continue
        if slope < 0 and x2 < WIDTH / 2 - CENTER_MARGIN:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > WIDTH / 2 + CENTER_MARGIN:
            right.append((x1, y1, x2, y2))
    return left, right


def get_pos(lines):
    if not lines:
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
    return int((GAP / 2 - b) / m)


def detect_lane(frame):
    """
    Returns:
        lpos  : 왼쪽 차선 x 위치 (없으면 None)
        rpos  : 오른쪽 차선 x 위치 (없으면 None)
        left  : 왼쪽 선분 목록
        right : 오른쪽 선분 목록
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    roi  = edge[OFFSET:OFFSET + GAP, 0:WIDTH]

    lines = cv2.HoughLinesP(roi, 1, math.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LEN,
                            maxLineGap=HOUGH_MAX_GAP)
    if lines is None:
        return None, None, [], []

    left, right = divide_left_right(lines)
    return get_pos(left), get_pos(right), left, right


# ====================================================================
# LiDAR 분석
# ====================================================================

def analyze_lidar(scan):
    """
    XY ROI 박스 기반 장애물 감지.
    Returns dict:
        front_min   : ROI 내 가장 가까운 장애물 거리
        front_angle : 장애물 x 좌표 부호 (오른쪽=+, 왼쪽=-)
        left_min    : 왼쪽 벽 최소 거리 (터널용)
        right_min   : 오른쪽 벽 최소 거리 (터널용)
        obstacle    : bool
        tunnel      : bool
    """
    if scan is None:
        return None

    offset_rad = math.radians(ANGLE_OFFSET_DEG)
    angle_min  = scan.angle_min
    angle_inc  = scan.angle_increment
    ranges     = scan.ranges

    roi_pts = []   # ROI 안 (x, y, dist)
    left_vals, right_vals = [], []

    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r) or r < 0.01:
            continue
        rad = angle_min + i * angle_inc + offset_rad
        x   = math.sin(rad) * r
        y   = math.cos(rad) * r

        # ROI 장애물 체크
        if ROI_X_MIN <= x <= ROI_X_MAX and ROI_Y_MIN <= y <= ROI_Y_MAX:
            roi_pts.append((x, y, r))

        # 터널 벽 감지 (각도 인덱스 기반 유지)
        deg = math.degrees(rad) % 360
        if deg > 360 - SIDE_HALF_DEG:
            left_vals.append(r)
        elif deg < SIDE_HALF_DEG:
            right_vals.append(r)

    # ROI 내 가장 가까운 포인트 + x spread
    front_min   = float('inf')
    front_x     = 0.0
    front_y     = ROI_Y_MIN
    x_coverage  = 0.0   # ROI x폭 중 장애물이 차지하는 비율 (0~1)

    if len(roi_pts) >= MIN_ROI_POINTS:
        closest   = min(roi_pts, key=lambda p: p[2])
        front_min = closest[2]
        front_x   = closest[0]
        front_y   = closest[1]

        xs         = [p[0] for p in roi_pts]
        x_spread   = max(xs) - min(xs)
        x_coverage = min(1.0, x_spread / (2.0 * ROI_X))

    left_min  = min(left_vals)  if left_vals  else float('inf')
    right_min = min(right_vals) if right_vals else float('inf')

    obstacle = len(roi_pts) >= MIN_ROI_POINTS
    tunnel   = (left_min < TUNNEL_DIST) and (right_min < TUNNEL_DIST) and not obstacle

    return {
        'front_min'  : front_min,
        'front_x'    : front_x,      # 가장 가까운 장애물 x (오른쪽+, 왼쪽-)
        'front_y'    : front_y,      # 가장 가까운 장애물 y (y_max=-0.1이 차량에 가장 가까움)
        'x_coverage' : x_coverage,   # ROI x폭 점유율 (0~1)
        'left_min'   : left_min,
        'right_min'  : right_min,
        'obstacle'   : obstacle,
        'tunnel'     : tunnel,
    }


# ====================================================================
# 주행 모드별 제어
# ====================================================================

def compute_angle_lane(lpos, rpos):
    """
    MODE 1 / LANE_BOTH / LANE_NONE:
    차선 정보만으로 중점 계산 → 조향각 반환
    EMA 반폭 갱신 포함.
    """
    global ema_half_width, prev_center

    if lpos is not None and rpos is not None:
        center = (lpos + rpos) // 2
        half = (rpos - lpos) / 2.0
        ema_half_width = (EMA_ALPHA * half + (1 - EMA_ALPHA) * ema_half_width
                          if ema_half_width is not None else half)
        mode_tag = "LANE_BOTH"

    elif lpos is not None:
        center = (int(lpos + ema_half_width)
                  if ema_half_width is not None else prev_center)
        mode_tag = "LANE_LEFT"

    elif rpos is not None:
        center = (int(rpos - ema_half_width)
                  if ema_half_width is not None else prev_center)
        mode_tag = "LANE_RIGHT"

    else:
        center = prev_center
        mode_tag = "LANE_NONE"

    prev_center = center
    angle = (center - WIDTH // 2) * GAIN
    return angle, center, mode_tag


def compute_angle_avoid(lpos, rpos, lidar_info):
    """
    MODE 2 – 장애물 회피.
    오프셋 크기는 세 요소로 결정:
      dist_factor   : y_max(-0.1)에 가까울수록 (차량에 가까울수록) 크게
      x_factor      : x가 0(중앙)에 가까울수록 크게, ROI 끝(±0.2)이면 작게
      coverage      : ROI x폭 중 장애물 점유율이 높을수록 크게
    """
    global prev_center

    front_x    = lidar_info['front_x']
    front_y    = lidar_info['front_y']
    x_coverage = lidar_info['x_coverage']

    # 1. 거리 요소: y_max에 가까울수록 1, y_min에 가까울수록 0
    roi_y_range  = ROI_Y_MAX - ROI_Y_MIN          # 0.5
    dist_factor  = (front_y - ROI_Y_MIN) / roi_y_range
    dist_factor  = max(0.0, min(1.0, dist_factor))

    # 2. 중앙 근접 요소: x=0이면 1, x=±ROI_X이면 0
    x_factor = 1.0 - abs(front_x) / ROI_X
    x_factor = max(0.0, min(1.0, x_factor))

    # 3. 점유율 요소: coverage 그대로 (0~1)

    # 오프셋 합산: 기본 80px + 각 요소 최대 80px씩 추가 → 최대 320px
    offset_px = int(AVOID_OFFSET_PX * (1.0 + dist_factor + x_factor + x_coverage))

    # 회피 방향: front_x >= 0 이면 direction = 1, < 0 이면 -1
    direction = 1 if front_x >= 0 else -1

    _, center, _ = compute_angle_lane(lpos, rpos)
    avoid_center = int(np.clip(center + direction * offset_px, 0, WIDTH - 1))
    prev_center  = avoid_center

    angle = (avoid_center - WIDTH // 2) * GAIN
    return angle, avoid_center, "AVOID"


def compute_angle_tunnel(lidar_info):
    """
    MODE 3 – 터널 (양쪽 차선 없음, 좌우 벽 존재):
    좌우 벽 거리 차이를 0으로 만드는 방향으로 조향 (벽 균등 추종).
    """
    left_min  = lidar_info['left_min']
    right_min = lidar_info['right_min']

    # 유효하지 않은 거리는 대칭값으로 보완
    if math.isinf(left_min):
        left_min = right_min
    if math.isinf(right_min):
        right_min = left_min

    # 오른쪽이 더 가까우면 왼쪽으로(음수), 왼쪽이 가까우면 오른쪽으로(양수)
    diff  = left_min - right_min   # 양수 = 오른쪽이 더 가까움 → 왼쪽으로
    angle = -diff * TUNNEL_GAIN    # 부호 반전: 좌로 틀어야 하므로

    return float(angle), WIDTH // 2, "TUNNEL"


# ====================================================================
# 디버그 표시
# ====================================================================

def draw_debug(frame, center, mode, left, right, lidar_info):
    y = OFFSET + GAP // 2
    cv2.rectangle(frame, (0, OFFSET), (WIDTH - 1, OFFSET + GAP), (0, 255, 0), 2)

    for x1, y1, x2, y2 in left:
        cv2.line(frame, (x1, y1 + OFFSET), (x2, y2 + OFFSET), (0, 0, 255), 2)
    for x1, y1, x2, y2 in right:
        cv2.line(frame, (x1, y1 + OFFSET), (x2, y2 + OFFSET), (255, 0, 0), 2)

    cv2.circle(frame, (center, y), 6, (0, 255, 255), -1)   # 중점 노랑
    cv2.circle(frame, (WIDTH // 2, y), 6, (255, 255, 255), -1)  # 화면 중심 흰

    color_map = {
        "AVOID"      : (0, 128, 255),
        "TUNNEL"     : (255, 165, 0),
        "LANE_LEFT"  : (200, 200, 0),
        "LANE_RIGHT" : (200, 200, 0),
        "LANE_BOTH"  : (0, 255, 0),
        "LANE_NONE"  : (100, 100, 100),
    }
    color = color_map.get(mode, (255, 255, 255))
    cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if lidar_info:
        info_str = "F:%.2f x:%.2f cov:%.0f%% L:%.2f R:%.2f" % (
            lidar_info['front_min'],
            lidar_info['front_x'],
            lidar_info['x_coverage'] * 100,
            lidar_info['left_min']  if not math.isinf(lidar_info['left_min'])  else 9.99,
            lidar_info['right_min'] if not math.isinf(lidar_info['right_min']) else 9.99,
        )
        cv2.putText(frame, info_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('obstacle_lane_drive', frame)
    cv2.waitKey(1)


# ====================================================================
# 메인 루프
# ====================================================================

def main():
    global motor_pub, drive_mode

    rospy.init_node('obstacle_lane_drive')
    motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    rospy.Subscriber('/scan', LaserScan, lidar_callback, queue_size=1)
    rospy.on_shutdown(lambda: drive(0, 0))

    rospy.sleep(2.0)
    print("[obstacle_lane_drive] started")

    count = 0
    rate  = rospy.Rate(30)

    while not rospy.is_shutdown():
        if image.size == 0:
            rate.sleep()
            continue

        frame = image.copy()

        # ── 차선 검출 ──────────────────────────────────────────────
        lpos, rpos, left_lines, right_lines = detect_lane(frame)

        # ── LiDAR 분석 ────────────────────────────────────────────
        lidar_info = analyze_lidar(lidar_scan)

        # ── 모드 결정 및 조향각 산출 ──────────────────────────────
        # 우선순위: AVOID > TUNNEL > LANE
        if lidar_info and lidar_info['obstacle']:
            # MODE 2: 장애물 회피
            angle, center, drive_mode = compute_angle_avoid(lpos, rpos, lidar_info)

        elif lidar_info and lidar_info['tunnel']:
            # MODE 3: 터널 (차선 없음 + 양쪽 벽)
            angle, center, drive_mode = compute_angle_tunnel(lidar_info)

        else:
            # MODE 1 / 일반 차선 주행
            angle, center, drive_mode = compute_angle_lane(lpos, rpos)

        # ── 클램프 및 발행 ────────────────────────────────────────
        angle = float(np.clip(angle, -50, 50))
        drive(angle, SPEED)

        # ── 디버그 ────────────────────────────────────────────────
        if SHOW_DEBUG:
            draw_debug(frame, center, drive_mode, left_lines, right_lines, lidar_info)

        count += 1
        if count % 30 == 0:
            half_str = "%.0f" % ema_half_width if ema_half_width is not None else "-"
            lidar_str = ""
            if lidar_info:
                lidar_str = " | F=%.2f L=%.2f R=%.2f" % (
                    lidar_info['front_min'],
                    lidar_info['left_min']  if not math.isinf(lidar_info['left_min'])  else 9.99,
                    lidar_info['right_min'] if not math.isinf(lidar_info['right_min']) else 9.99,
                )
            print("mode=%-12s center=%3d angle=%4d ema_half=%s%s"
                  % (drive_mode, center, int(angle), half_str, lidar_str))

        rate.sleep()


if __name__ == '__main__':
    main()
