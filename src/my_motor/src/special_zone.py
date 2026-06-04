#!/usr/bin/env python
# -*- coding: utf-8 -*-

# special_zone.py - 특수구역(횡단보도/빗금) 감지 전용 프로세서
#   ImageProcessor/LidarProcessor와 동일하게 cfg를 주입받는 순수 감지 클래스.
#   상태머신/타이머/카운터는 XycarController가 소유하고, 여기는 감지만 담당.
#
#   횡단보도(CROSSWALK): 어두운 트랙 위에 같은 y높이로 가로로 늘어선 흰 블록 N개 이상
#   빗금(HATCH)        : 어두운 영역 내부를 채운 흰 사선 비율이 임계 이상

import numpy as np
import cv2


class SpecialZoneProcessor:
    def __init__(self, cfg):
        self.cfg = cfg   # 가변 감지 상태를 들지 않음 (순수 함수형)

    def get_detect_roi(self, frame):
        return frame[self.cfg.detect_roi_top:self.cfg.detect_roi_bottom,
                     self.cfg.detect_roi_left:self.cfg.detect_roi_right]

    def detect_crosswalk(self, frame):
        """ROI의 흰 블록 중 '같은 y높이에 가로로 퍼진' 블록이 N개 이상이고
        그 블록들이 어두운 트랙 위에 있으면 횡단보도. -> (detected, block_rects, white_mask)"""
        roi  = self.get_detect_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, white = cv2.threshold(gray, self.cfg.cross_white_thresh, 255, cv2.THRESH_BINARY)
        _, dark  = cv2.threshold(gray, self.cfg.cross_dark_thresh,  255, cv2.THRESH_BINARY_INV)
        dark_d   = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

        n, _, stats, cent = cv2.connectedComponentsWithStats(white, 8)
        blobs = []
        H, W  = gray.shape
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.cfg.cross_blob_min_area or area > self.cfg.cross_blob_max_area:
                continue
            if w > self.cfg.cross_blob_max_w or h > self.cfg.cross_blob_max_h:
                continue
            cx, cy = int(cent[i][0]), int(cent[i][1])
            if dark_d[min(cy, H - 1), min(cx, W - 1)] == 0:
                continue
            blobs.append((cent[i][0], cent[i][1], x, y, w, h))

        best_row = []
        if len(blobs) >= self.cfg.cross_min_blocks:
            for yc in range(0, H, 8):
                row = [b for b in blobs if abs(b[1] - yc) <= self.cfg.cross_row_band]
                if len(row) >= self.cfg.cross_min_blocks:
                    xs = [b[0] for b in row]
                    if max(xs) - min(xs) >= self.cfg.cross_row_span and len(row) > len(best_row):
                        best_row = row

        block_rects = [(int(b[2]), int(b[3]), int(b[4]), int(b[5])) for b in best_row]
        detected    = (len(best_row) >= self.cfg.cross_min_blocks)
        return detected, block_rects, white

    def detect_hatch(self, frame):
        """ROI에서 어두운 영역을 CLOSE+ERODE 해 '내부'를 구한 뒤, 그 내부의 흰 픽셀
        비율이 임계 이상이면 빗금. -> (detected, stripe_rects, white_pct)"""
        roi  = self.get_detect_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        dark = (gray < self.cfg.hatch_dark_thresh).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        inside = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        area_inside = int(np.sum(inside > 0))
        if area_inside < self.cfg.hatch_inside_min_area:
            return False, [], 0.0

        white     = ((gray > self.cfg.hatch_white_thresh) & (inside > 0)).astype(np.uint8)
        white_pct = 100.0 * int(np.sum(white > 0)) / float(area_inside)

        stripe_rects = []
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a >= 15:
                stripe_rects.append((x, y, w, h))

        detected = (white_pct >= self.cfg.hatch_white_pct)
        return detected, stripe_rects, white_pct

    def draw_overlay(self, frame, cross_blocks, hatch_rects, drive_state,
                     stop_remain, hatch_pct, hatch_delay_remain):
        """특수구역 관련 오버레이만 그림. imshow/waitKey는 호출하지 않음
        (창 flush는 ImageProcessor.draw_debug가 단독 담당). 호출 순서: 이 함수 -> draw_debug."""
        L, T = self.cfg.detect_roi_left, self.cfg.detect_roi_top

        # 감지 ROI 박스 (마젠타)
        cv2.rectangle(frame, (L, T),
                      (self.cfg.detect_roi_right, self.cfg.detect_roi_bottom),
                      (255, 0, 255), 1)

        # 횡단보도 블록 (초록 박스)
        for (x, y_b, w, h) in cross_blocks:
            cv2.rectangle(frame, (x + L, y_b + T), (x + L + w, y_b + T + h), (0, 255, 0), 2)

        # 빗금 흰 줄무늬 (빨강 박스)
        for (x, y_b, w, h) in hatch_rects:
            cv2.rectangle(frame, (x + L, y_b + T), (x + L + w, y_b + T + h), (0, 0, 255), 1)

        # 주행 상태 라벨 (좌상단)
        state_color = {
            "DRIVE":          (0, 255, 0),
            "CROSSWALK_STOP": (0, 165, 255),
            "HATCH_STOP":     (0, 0, 255),
        }
        color = state_color.get(drive_state, (255, 255, 255))
        label = drive_state
        if drive_state == "CROSSWALK_STOP" and stop_remain > 0:
            label = "CROSSWALK_STOP (%.1fs)" % stop_remain
        cv2.putText(frame, label, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 빗금 비율 / 빗금 정지 대기
        hatch_info = "hatch=%.1f%%" % hatch_pct
        if hatch_delay_remain > 0:
            hatch_info += "  [HATCH in %.1fs]" % hatch_delay_remain
        cv2.putText(frame, hatch_info, (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
