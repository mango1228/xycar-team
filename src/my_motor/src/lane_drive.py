#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lane_drive.py - 허프변환 기반 차선주행 노드
# 베이스: auto_drive/hough_drive.py 구조 (ROI 먼저 자르고 Hough)
# 개선: 튜닝 Canny / 조향 클램프+게인 / 깔끔한 종료
#       한쪽 차선만 보일 때 -> 보이는 차선에서 중점을 직접 재구성 (반폭 EMA 사용)
#       라이다: lidar_only와 동일 방식 (laser_frame, cos/sin 표준 좌표)

import rospy
import numpy as np
import cv2, math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as GeoPoint
from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers



class ImageProcessor:
    def __init__ (self, cfg):
        self.cfg = cfg
        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.ema_half_width = None
        self.prev_center= self.cfg.width // 2

        rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback)
        
    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def divide_left_right(self, lines):
        left, right = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = float(y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.1 or abs(slope) > 10:
                continue
            if slope < 0 and x2 < self.cfg.width/2 - self.cfg.center_margin:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > self.cfg.width/2 + self.cfg.center_margin:
                right.append((x1, y1, x2, y2))
        return left, right
    
    def get_pos(self, lines):
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
        return int((self.cfg.gap/2 - b) / m)
    
    def process(self, frame):
        """프레임 -> (center, mode, 좌/우 선분 리스트, lpos, rpos, corner)"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)

        roi   = edge[self.cfg.offset:self.cfg.offset+self.cfg.gap, 0:self.cfg.width]
        lines = cv2.HoughLinesP(roi, 1, math.pi/180, self.cfg.hough_threshold,
                                minLineLength=self.cfg.hough_min_len, maxLineGap=self.cfg.hough_max_gap)

        if lines is None:
            return self.prev_center, "NONE", [], [], None, None, "STRAIGHT", None

        left, right = self.divide_left_right(lines)
        # 여러 선분 중 무게중심이 가장 안쪽(중앙에 가까운) 1개만 채택 -> 옆 차선/노이즈 배제
        mid = lambda l: (l[0] + l[2]) / 2.0
        if left:
            left = [max(left, key=mid)]    # 왼쪽: 무게중심이 가장 오른쪽인 선분
        if right:
            right = [min(right, key=mid)]  # 오른쪽: 무게중심이 가장 왼쪽인 선분
        lpos = self.get_pos(left)
        rpos = self.get_pos(right)

        # 코너 판단: 왼쪽 차선 기울기만 사용 (이미지 좌표, 직진이어도 음수). 좌회전만 판단.
        corner   = "STRAIGHT"
        left_slope = None
        if left:
            left_slope = float(left[0][3] - left[0][1]) / float(left[0][2] - left[0][0])
            dev = left_slope - self.cfg.corner_left_base   # 기준값 대비 편차 (좌회전이면 더 음수)
            if dev < -self.cfg.corner_slope_thresh:
                corner = "LEFT"           # 기울기 가팔라짐(더 음수) -> 좌회전

        if lpos is not None and rpos is not None:
            center = (lpos + rpos) // 2
            half = (rpos - lpos) / 2.0
            if self.ema_half_width is None:
                self.ema_half_width = half
            else:
                self.ema_half_width = self.cfg.ema_alpha * half + (1 - self.cfg.ema_alpha) * self.ema_half_width
            mode = "BOTH"
        elif lpos is not None:
            # 왼쪽만 보임 -> 기억한 반폭의 ONE_LANE_RATIO 만큼 오른쪽으로
            center = int(lpos + self.ema_half_width * self.cfg.one_lane_ratio) if self.ema_half_width is not None else self.prev_center
            mode = "LEFT"
        elif rpos is not None:
            # 오른쪽만 보임 -> 기억한 반폭의 ONE_LANE_RATIO 만큼 왼쪽으로
            center = int(rpos - self.ema_half_width * self.cfg.one_lane_ratio) if self.ema_half_width is not None else self.prev_center
            mode = "RIGHT"
        else:
            center = self.prev_center
            mode = "NONE"

        # 코너 보정 (좌회전 시 추종점 왼쪽으로)
        if corner == "LEFT":
            center -= self.cfg.corner_shift_px

        center = max(0, min(self.cfg.width - 1, center))
        self.prev_center = center
        return center, mode, left, right, lpos, rpos, corner, left_slope


    def draw_debug(self, frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos, corner, left_slope):
        y = self.cfg.offset + self.cfg.gap // 2
        cv2.rectangle(frame, (0, self.cfg.offset), (self.cfg.width-1, self.cfg.offset+self.cfg.gap), (0, 255, 0), 2)
        for x1, y1, x2, y2 in left:
            cv2.line(frame, (x1, y1+self.cfg.offset), (x2, y2+self.cfg.offset), (0, 0, 255), 2)
        for x1, y1, x2, y2 in right:
            cv2.line(frame, (x1, y1+self.cfg.offset), (x2, y2+self.cfg.offset), (255, 0, 0), 2)
        # 차선 검출 위치 (get_pos 반환값) 검정 점
        if lpos is not None:
            cv2.circle(frame, (lpos, y), 6, (0, 0, 0), -1)
        if rpos is not None:
            cv2.circle(frame, (rpos, y), 6, (0, 0, 0), -1)
        cv2.circle(frame, (cam_center, y), 6, (255, 128, 0),   -1)
        if lidar_c is not None:
            cv2.circle(frame, (lidar_c, y), 6, (0, 0, 255),    -1)
        cv2.circle(frame, (center, y),     8, (0, 255, 255),    2)
        cv2.circle(frame, (self.cfg.width//2, y),   6, (255, 255, 255), -1)
        cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # 왼쪽 차선 기울기 값 표시 (좌상단)
        slope_str = "Lslope: %.2f" % left_slope if left_slope is not None else "Lslope: -"
        cv2.putText(frame, slope_str, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # 코너 판단 글씨 (화면 하단 중앙)
        label = {"STRAIGHT": "STRAIGHT", "LEFT": "<< LEFT"}.get(corner, corner)
        color = (0, 255, 0) if corner == "STRAIGHT" else (0, 165, 255)
        cv2.putText(frame, label, (self.cfg.width//2 - 60, self.cfg.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('lane_drive', frame)
        cv2.waitKey(1)

    def shutdown(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass


class LidarProcessor:
    def __init__ (self, cfg):
        self.cfg = cfg

        self.roi_ang_min = math.atan2(self.cfg.lidar_roi_y_max, -self.cfg.lidar_roi_x)  # ≈ -135°
        self.roi_ang_max = math.atan2(self.cfg.lidar_roi_y_max, self.cfg.lidar_roi_x)   # ≈  -45°
        self.roi_ang_center = (self.roi_ang_min + self.roi_ang_max) / 2.0        # ≈  -90° (차량 전방)

        self.lidar_scan = None

        rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=1)
        self.marker_pub = rospy.Publisher('/lane_drive/roi_markers', MarkerArray, queue_size=1)


    def lidar_callback(self, data):
        self.lidar_scan = data

    def set_roi(self, x, y_min, y_max):
        """라이다 ROI를 런타임에 변경. 각도 경계도 함께 재계산해야
        get_gaps / 원뿔 마커가 새 ROI를 따라간다."""
        self.cfg.lidar_roi_x     = x
        self.cfg.lidar_roi_y_min = y_min
        self.cfg.lidar_roi_y_max = y_max
        self.roi_ang_min = math.atan2(y_max, -x)
        self.roi_ang_max = math.atan2(y_max,  x)
        self.roi_ang_center = (self.roi_ang_min + self.roi_ang_max) / 2.0

    def get_roi_data(self, scan):
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
            if (-self.cfg.lidar_roi_x <= x <= self.cfg.lidar_roi_x and
                    self.cfg.lidar_roi_y_min <= y <= self.cfg.lidar_roi_y_max):
                data.append((x, y, rad))
        return data

    def get_gaps(self,data):
        """ROI 안 포인트 각도 사이 빈 구간 [(a1, a2)] 반환.
        포인트 없으면 ROI 전체가 하나의 빈 공간. MIN_GAP_ANG 미만은 무시."""
        if not data:
            return [(self.roi_ang_min, self.roi_ang_max)]
        angles = sorted(d[2] for d in data)
        angles = [a for a in angles if self.roi_ang_min <= a <= self.roi_ang_max]
        if not angles:
            return [(self.roi_ang_min, self.roi_ang_max)]
        raw = ([(self.roi_ang_min, angles[0])] +
            [(angles[i], angles[i + 1]) for i in range(len(angles) - 1)] +
            [(angles[-1], self.roi_ang_max)])
        return [(a1, a2) for a1, a2 in raw if a2 - a1 >= self.cfg.min_gap_ang]


    def publish_roi_markers(self, data, gaps, bisector, scan):
        """RViz 마커 발행: 초록 박스 / 빨강 포인트 / 하늘색 원뿔 / 노란 화살표"""
        if scan is None:
            return
        stamp    = scan.header.stamp
        frame_id = scan.header.frame_id
        arr      = MarkerArray()

        # ── 1. ROI 박스 (초록 LINE_STRIP) ──────────────────────────────
        box = Marker()
        box.header.stamp = stamp; box.header.frame_id = frame_id
        box.ns = "roi"; box.id = 0; box.type = Marker.LINE_STRIP
        box.action = Marker.ADD
        box.pose.orientation.w = 1.0
        box.scale.x = 0.02
        box.color.r = 0.0; box.color.g = 1.0; box.color.b = 0.0; box.color.a = 1.0
        box.lifetime = rospy.Duration(0.1)
        for cx, cy in [(-self.cfg.lidar_roi_x, self.cfg.lidar_roi_y_min),
                    (self.cfg.lidar_roi_x, self.cfg.lidar_roi_y_min),
                    (self.cfg.lidar_roi_x, self.cfg.lidar_roi_y_max),
                    (-self.cfg.lidar_roi_x, self.cfg.lidar_roi_y_max),
                    (-self.cfg.lidar_roi_x, self.cfg.lidar_roi_y_min)]:
            box.points.append(self.pt(cx, cy))
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
            pm.points.append(self.pt(x, y))
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
        origin = self.pt(0.0, 0.0)
        for a1, a2 in gaps:
            n = max(2, int((a2 - a1) / 0.05))
            for k in range(n):
                ang0 = a1 + (a2 - a1) * k / n
                ang1 = a1 + (a2 - a1) * (k + 1) / n
                p0 = self.pt(math.cos(ang0) * self.cfg.r_viz, math.sin(ang0) * self.cfg.r_viz)
                p1 = self.pt(math.cos(ang1) * self.cfg.r_viz, math.sin(ang1) * self.cfg.r_viz)
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
                            self.pt(math.cos(bisector) * 0.5, math.sin(bisector) * 0.5)]
        else:
            arrow.action = Marker.DELETE
        arr.markers.append(arrow)

        self.marker_pub.publish(arr)

    def pt(self, x, y, z=0.0):
        p = GeoPoint(); p.x = x; p.y = y; p.z = z; return p


class XycarDriver:
    def __init__(self):
        self.motor_pub  = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.motor_msg  = xycar_motor()   

    def drive(self, angle, speed):
        self.motor_msg.angle = int(angle)
        self.motor_msg.speed = int(speed)
        self.motor_pub.publish(self.motor_msg)

    def shutdown(self):
        rospy.loginfo("Shutting down...")
        self.drive(0, 0)


class LaneFollower:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lidar_c = None
        self.bisector = None

    def correct_lane(self, gaps, roi_ang_center):
        largest  = max(gaps, key=lambda g: g[1] - g[0])
        self.bisector = (largest[0] + largest[1]) / 2.0
        # ROI_ANG_CENTER - bisector: laser_frame에서 car left/right 방향 보정
        # bisector > CENTER(=-90°) → car's left → negative offset → steer left
        dev = roi_ang_center - self.bisector
        self.lidar_c = max(0, min(self.cfg.width - 1, int(self.cfg.width // 2 + dev * self.cfg.lidar_center_gain)))
        return self.lidar_c, self.bisector

    def correct_lane_leftmost(self, gaps, roi_ang_center, min_deg=15.0, gain_scale=1.0):
        """min_deg 이상 벌어진 부채꼴 중 '가장 왼쪽'(bisector가 가장 작은=차량 좌측) 것을 추종.
        gain_scale: 추종 계수(lidar_center_gain) 배율 (더 공격적으로 추종).
        조건 만족하는 부채꼴이 없으면 기존 최대 부채꼴 방식으로 폴백.
        실차 확인 결과 left = bisector 최소(min). 반대로 가면 min → max 로 되돌릴 것."""
        min_rad = math.radians(min_deg)
        wide = [g for g in gaps if (g[1] - g[0]) >= min_rad]
        if not wide:
            return self.correct_lane(gaps, roi_ang_center)
        leftmost = min(wide, key=lambda g: (g[0] + g[1]) / 2.0)
        self.bisector = (leftmost[0] + leftmost[1]) / 2.0
        dev = roi_ang_center - self.bisector
        gain = self.cfg.lidar_center_gain * gain_scale
        self.lidar_c = max(0, min(self.cfg.width - 1, int(self.cfg.width // 2 + dev * gain)))
        return self.lidar_c, self.bisector
    

class ARtagDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.callback, queue_size=1)
        self.detected = False     # 이번 프레임에 ROI 안에서 마커가 보였나
        self.marker_id = None
        self.distance = None      # 카메라 전방 거리(z, m)
        self.pixel = None         # 검출 마커의 화면 픽셀 (u, v) — 디버그용

    def callback(self, msg):      # ← 메시지 인자 필수
        # 마커 3D pose(미터, 카메라 광학좌표 x오른쪽/y아래/z전방)를 픽셀로 투영해서
        # AR 전용 ROI 박스(화면 하단 40%, 좌우 전체) 안에 들어오는 마커만 인정한다.
        best = None  # (z, id, u, v) — ROI 안에서 가장 가까운(z 최소) 마커
        for m in msg.markers:
            p = m.pose.pose.position
            z = p.z
            if z <= 0:            # 카메라 뒤/평면 마커 무시 (0 나눗셈 방지)
                continue
            u = self.cfg.cam_fx * p.x / z + self.cfg.cam_cx
            v = self.cfg.cam_fy * p.y / z + self.cfg.cam_cy
            if (self.cfg.ar_roi_left <= u <= self.cfg.ar_roi_right and
                    self.cfg.ar_roi_top <= v <= self.cfg.ar_roi_bottom):
                if best is None or z < best[0]:
                    best = (z, m.id, u, v)

        if best is None:          # 마커가 없거나 모두 ROI 밖
            self.detected = False
            self.pixel = None
            return

        self.detected  = True
        self.distance  = best[0]
        self.marker_id = best[1]
        self.pixel     = (int(best[2]), int(best[3]))