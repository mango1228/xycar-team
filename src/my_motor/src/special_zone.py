#!/usr/bin/env python
# -*- coding: utf-8 -*-

# special_zone.py - 특수구역(횡단보도/빗금) 판정 전용 프로세서
#   cfg를 주입받는 순수 판정 클래스. 상태머신/타이머는 XycarController가 소유.
#
#   판정: 감지 ROI '어두운 영역 내부'의 흰 비율이 classify_white_pct 이상이면,
#         내부 흰 무늬 선분들의 (길이 가중) 평균 각도로 구분 (0도=수평, 90도=수직).
#         평균각 >= classify_angle_deg → 횡단보도(세로) / < → 빗금(사선)

import math
import numpy as np
import cv2


class SpecialZoneProcessor:
    def __init__(self, cfg):
        self.cfg = cfg   # 가변 상태 없음 (순수 함수형)

    def get_detect_roi(self, frame):
        return frame[self.cfg.detect_roi_top:self.cfg.detect_roi_bottom,
                     self.cfg.detect_roi_left:self.cfg.detect_roi_right]

    def classify_zone(self, frame):
        """ROI 어두운 영역 내부 흰 비율 >= classify_white_pct 면, 내부 선분 평균각으로
        횡단보도(가로) / 빗금(사선) 판정.
        return (verdict, white_pct, avg_angle, segments)
          verdict : 'none' | 'crosswalk' | 'hatch'
          segments: ROI 좌표계 (x1,y1,x2,y2) 리스트 (디버그용)"""
        roi  = self.get_detect_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 어두운 영역의 '내부' (CLOSE로 메우고 ERODE로 가장자리 깎음)
        dark = (gray < self.cfg.hatch_dark_thresh).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        inside = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        area_inside = int(np.sum(inside > 0))
        if area_inside < self.cfg.hatch_inside_min_area:
            return 'none', 0.0, None, []

        white = ((gray > self.cfg.hatch_white_thresh) & (inside > 0)).astype(np.uint8) * 255
        white_pct = 100.0 * int(np.sum(white > 0)) / float(area_inside)
        if white_pct < self.cfg.classify_white_pct:
            return 'none', white_pct, None, []

        # 내부 흰 무늬 선분 검출 → 각도(0=수평, 90=수직) 길이 가중 평균
        edges = cv2.Canny(white, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, self.cfg.stripe_hough_threshold,
                                minLineLength=self.cfg.stripe_min_len,
                                maxLineGap=self.cfg.stripe_max_gap)
        if lines is None:
            return 'none', white_pct, None, []

        ang_sum = 0.0
        len_sum = 0.0
        segs = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            ang = abs(math.degrees(math.atan2(dy, dx)))
            if ang > 90.0:
                ang = 180.0 - ang          # 0~90 으로 접기 (/ 와 \ 모두 사선으로)
            length = math.hypot(dx, dy)
            ang_sum += ang * length        # 길이 가중 (긴 모서리가 방향을 지배)
            len_sum += length
            segs.append((x1, y1, x2, y2))

        if len_sum == 0:
            return 'none', white_pct, None, []

        avg_angle = ang_sum / len_sum
        # 횡단보도=세로(각도 큼, ~90), 빗금=사선(~45). 기준각 이상이면 횡단보도
        verdict = 'crosswalk' if avg_angle >= self.cfg.classify_angle_deg else 'hatch'
        return verdict, white_pct, avg_angle, segs

    def draw_classify_debug(self, frame, verdict, white_pct, avg_angle, segments):
        """검출 노드(special_zone_detector) 전용 디버그 창. imshow/waitKey 직접 호출."""
        L, T = self.cfg.detect_roi_left, self.cfg.detect_roi_top

        # 감지 ROI 박스 (마젠타)
        cv2.rectangle(frame, (L, T),
                      (self.cfg.detect_roi_right, self.cfg.detect_roi_bottom),
                      (255, 0, 255), 1)

        # 검출된 선분 (노랑)
        for (x1, y1, x2, y2) in segments:
            cv2.line(frame, (x1 + L, y1 + T), (x2 + L, y2 + T), (0, 255, 255), 2)

        # 판정 텍스트 (횡단보도=초록 / 빗금=빨강 / none=회색)
        color = {'crosswalk': (0, 255, 0), 'hatch': (0, 0, 255)}.get(verdict, (200, 200, 200))
        ang_str = "%.0f" % avg_angle if avg_angle is not None else "-"
        cv2.putText(frame, "%s  white=%.1f%%  ang=%s" % (verdict.upper(), white_pct, ang_str),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('special_zone_detector', frame)
        cv2.waitKey(1)
