#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lane_drive.py - 허프변환 기반 차선주행 노드
# 베이스: auto_drive/hough_drive.py 구조 (ROI 먼저 자르고 Hough)
# 개선: 튜닝 Canny / 조향 클램프+게인 / 깔끔한 종료
#       한쪽 차선만 보일 때 -> 보이는 차선에서 중점을 직접 재구성 (반폭 EMA 사용)

import os
import rospy
import numpy as np
import cv2, math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor

# ===== 튜닝 파라미터 (canny_tune으로 검증) =====
CANNY_LOW  = 40      # Canny 아래 임계값
CANNY_HIGH = 100     # Canny 위 임계값
OFFSET     = 340     # ROI 띠 시작 row
GAP        = 40      # ROI 띠 높이
GAIN       = 0.4     # 조향 P게인 (작을수록 둔감 -> 흔들림 적음)
SPEED      = 5       # 주행 속도 (0~5)
EMA_ALPHA  = 0.3     # 반폭 EMA 계수 (클수록 최신값에 민감)
SHOW_DEBUG = True    # 디버그 창 표시 (헤드리스 실행이면 False)

# 디스플레이 없으면 디버그 창 자동 끔 (no-display에서 cv2.imshow가 segfault 내는 것 방지)
if SHOW_DEBUG and not os.environ.get('DISPLAY'):
    print("[lane_drive] DISPLAY 없음 - 디버그 창 비활성화 (창 보려면 ssh -Y 로 접속)")
    SHOW_DEBUG = False

WIDTH, HEIGHT = 640, 480

# HoughLinesP 파라미터
HOUGH_THRESHOLD = 30
HOUGH_MIN_LEN   = 20
HOUGH_MAX_GAP   = 10

CENTER_MARGIN = 90   # 좌/우 분리 시 중앙 여유 (이 안쪽 선은 무시)

image = np.empty(shape=[0])
bridge = CvBridge()
motor_pub = None
motor_msg = xycar_motor()

# 한쪽 차선 처리용 상태
ema_half_width = None        # 반(half) 차선폭의 EMA. 양쪽 차선 처음 보면 설정됨
prev_center    = WIDTH // 2  # 직전 중점 (검출 실패 시 유지)


def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")


def drive(angle, speed):
    motor_msg.angle = int(angle)
    motor_msg.speed = int(speed)
    motor_pub.publish(motor_msg)


def divide_left_right(lines):
    """Hough 선분들을 기울기/위치로 좌/우 차선으로 분리"""
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue                                 # 수직선 제외
        slope = float(y2 - y1) / float(x2 - x1)
        if abs(slope) < 0.1 or abs(slope) > 10:
            continue                                 # 수평/극단 기울기 제외
        if slope < 0 and x2 < WIDTH/2 - CENTER_MARGIN:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > WIDTH/2 + CENTER_MARGIN:
            right.append((x1, y1, x2, y2))
    return left, right


def get_pos(lines):
    """선분 평균 직선으로 ROI 띠 중앙(y=GAP/2)에서의 x좌표 산출. 없으면 None"""
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
    """프레임 -> (center, mode, 좌/우 선분 리스트)
       mode: BOTH(양쪽) / LEFT(왼쪽만) / RIGHT(오른쪽만) / NONE(없음)"""
    global ema_half_width, prev_center

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    roi = edge[OFFSET:OFFSET+GAP, 0:WIDTH]           # ROI 먼저 -> 좌표는 띠 내부 기준
    lines = cv2.HoughLinesP(roi, 1, math.pi/180, HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)

    if lines is None:
        return prev_center, "NONE", [], []

    left, right = divide_left_right(lines)
    lpos = get_pos(left)
    rpos = get_pos(right)

    if lpos is not None and rpos is not None:
        # 양쪽 다: 중점 = 가운데. 반폭(half width)을 EMA로 갱신
        center = (lpos + rpos) // 2
        half = (rpos - lpos) / 2.0
        if ema_half_width is None:
            ema_half_width = half
        else:
            ema_half_width = EMA_ALPHA * half + (1 - EMA_ALPHA) * ema_half_width
        mode = "BOTH"

    elif lpos is not None:
        # 왼쪽만 보임 -> 중점을 "왼쪽차선 + 반폭"으로 직접 계산
        if ema_half_width is None:
            center = prev_center                     # 아직 폭을 한 번도 못 잼
        else:
            center = int(lpos + ema_half_width)
        mode = "LEFT"

    elif rpos is not None:
        # 오른쪽만 보임 -> 중점을 "오른쪽차선 - 반폭"으로 직접 계산
        if ema_half_width is None:
            center = prev_center
        else:
            center = int(rpos - ema_half_width)
        mode = "RIGHT"

    else:
        # 둘 다 없음 -> 직전 중점 유지
        center = prev_center
        mode = "NONE"

    prev_center = center
    return center, mode, left, right


def draw_debug(frame, center, mode, left, right):
    y = OFFSET + GAP // 2
    cv2.rectangle(frame, (0, OFFSET), (WIDTH-1, OFFSET+GAP), (0, 255, 0), 2)
    for x1, y1, x2, y2 in left:                      # 검출 선분 (띠 좌표 -> 원본)
        cv2.line(frame, (x1, y1+OFFSET), (x2, y2+OFFSET), (0, 0, 255), 2)
    for x1, y1, x2, y2 in right:
        cv2.line(frame, (x1, y1+OFFSET), (x2, y2+OFFSET), (255, 0, 0), 2)
    cv2.circle(frame, (center, y),     6, (0, 255, 255),   -1)  # 중점 노랑
    cv2.circle(frame, (WIDTH//2, y),   6, (255, 255, 255), -1)  # 화면중심 흰
    cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('lane_drive', frame)
    cv2.waitKey(1)


def main():
    global motor_pub
    rospy.init_node('lane_drive')
    motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    rospy.on_shutdown(lambda: drive(0, 0))           # 종료 시 정지 (killall 안 씀)

    rospy.sleep(2.0)                                 # 카메라 준비 대기
    print("lane_drive started")

    count = 0
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if image.size == 0:
            rate.sleep()
            continue

        frame = image.copy()
        center, mode, left, right = process(frame)

        angle = (center - WIDTH // 2) * GAIN         # P 제어
        angle = max(-50, min(50, angle))             # ±50 클램프 -> 흔들림 억제

        drive(angle, SPEED)

        if SHOW_DEBUG:
            draw_debug(frame, center, mode, left, right)

        count += 1
        if count % 30 == 0:
            half_str = "%.0f" % ema_half_width if ema_half_width is not None else "-"
            print("mode=%s center=%d angle=%d ema_half=%s"
                  % (mode, center, int(angle), half_str))

        rate.sleep()


if __name__ == '__main__':
    main()
