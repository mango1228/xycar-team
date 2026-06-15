#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version0_2_lane_bag_two_sections_visual_csv.py

목적:
    하나의 rosbag에서 사용자가 코드 안에 지정한 여러 시간 구간만 골라서,
    버전 0.2 차선 검출 알고리즘을 검증한다.
    버전 0.2는 CLAHE/adaptive Canny 없이 local ROI 우선 탐색만 적용한다.

특징:
    - 원본 BAG은 자르지 않음
    - 코드 상단 TIME_SEGMENTS에 지정한 구간만 처리
    - 구간별 CSV 파일을 각각 생성
    - mp4 저장 없음
    - png/frame image 저장 없음
    - 실행 중 OpenCV 창으로 시각화 가능

수정 위치:
    아래 TIME_SEGMENTS의 start_sec, end_sec만 실제 BAG 확인 후 수정하면 된다.

기준 시간:
    start_sec, end_sec는 기본적으로 "해당 이미지 토픽의 첫 메시지 기준 상대시간 [s]"이다.
    예: start_sec=12.5, end_sec=19.8 이면 bag의 /usb_cam/image_raw 첫 프레임 후 12.5~19.8초 구간.

실행 예시:
    source /opt/ros/noetic/setup.bash

    python3 version0_2_lane_bag_two_sections_visual_csv.py \
      --bag /home/kim/smo_term/xycar-team/src/my_motor/src/2026-06-08-15-10-10.bag \
      --output-dir /home/kim/smo_term/xycar-team/src/my_motor/src/lane_bag_analysis_sections \
      --topic /usb_cam/image_raw \
      --view overlay \
      --delay-ms 80

키 조작:
    q 또는 ESC : 종료
    SPACE      : 일시정지/재개
    n          : 일시정지 상태에서 다음 프레임 1개 진행
"""

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import rosbag
    from cv_bridge import CvBridge
except Exception as exc:
    print("[ERROR] ROS 관련 모듈 import 실패")
    print("        먼저 ROS 환경을 source 하세요.")
    print("        예: source /opt/ros/noetic/setup.bash")
    print("        원인: {}".format(exc))
    sys.exit(1)


# ============================================================
# 사용자가 수정할 부분
# ============================================================
# start_sec, end_sec는 /usb_cam/image_raw 첫 프레임 기준 상대시간 [초]
# 정확한 시간은 bag 확인 후 여기만 바꾸면 된다.
# TIME_SEGMENTS = [
#     {
#         "name": "lap1_problem_section_v0_2",
#         "start_sec": 14.0,
#         "end_sec": 80.0,
#     },
#     {
#         "name": "lap2_problem_section_v0_2",
#         "start_sec": 104.0,
#         "end_sec": 115.0,
#     },
# ]
TIME_SEGMENTS = [
    {
        "name": "lap1_problem_section_v0_1",
        "start_sec": 37.440,
        "end_sec": 43.640,
    },
    {
        "name": "lap2_problem_section_v0_1",
        "start_sec": 105.440,
        "end_sec": 111.640,
    },
]

@dataclass
class BaselineConfig:
    # 기존 my_motor config.py / xycar_drive.launch 기준값
    width: int = 640
    height: int = 480

    canny_low: int = 40
    canny_high: int = 100

    offset: int = 330
    gap: int = 110

    ema_alpha: float = 0.3
    one_lane_ratio: float = 1.0

    corner_left_base: float = -0.75
    corner_slope_thresh: float = 0.4
    corner_shift_px: int = 70

    hough_threshold: int = 30
    hough_min_len: int = 20
    hough_max_gap: int = 10

    center_margin: int = 90

    # 분석용 판정 기준
    center_jump_threshold: int = 80
    lane_width_jump_threshold: int = 100

    # Version 0.2 local ROI parameters
    # full ROI 검출이 연속으로 실패한 뒤에만 last_good 주변 local ROI 복구를 시도한다.
    recovery_fail_threshold: int = 3
    recovery_window_px: int = 90
    recovery_pos_gate_px: int = 100
    recovery_lane_width_gate_px: int = 130
    recovery_clahe_clip: float = 2.0
    recovery_canny_sigma: float = 0.33
    recovery_canny_low_min: int = 15
    recovery_canny_high_min: int = 45
    recovery_canny_high_max: int = 180


class BaselineLaneDetector:
    """
    Version 0.2 detector.

    동작 원리:
        1) 매 프레임 baseline full ROI 검출을 먼저 계산해 baseline_mode로 기록한다.
        2) last_good_lpos/rpos가 있으면 해당 위치 주변 local ROI를 우선 탐색한다.
        3) local ROI에는 baseline과 같은 고정 Canny threshold + HoughLinesP를 적용한다.
        4) local 후보가 slope sign, 위치 gate, lane width 일관성을 만족하면 최종 mode로 채택한다.
        5) local ROI가 실패하거나 last_good이 없으면 full ROI 결과로 fallback한다.

    주의:
        Version 0.2는 prediction/last_good 기반 local-first 구조이므로,
        곡선 진입처럼 실제 차선 위치가 prediction과 달라지는 상황에서는 stale ROI 위험이 있다.
        이 한계는 Version 3에서 prediction mismatch를 full ROI 재탐색 트리거로 다루며 보완한다.
    """

    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.prev_center = cfg.width // 2
        self.ema_half_width: Optional[float] = None
        self.prev_lane_width: Optional[float] = None
        self.fail_count = 0

        # Version 0.2 state: 신뢰도 높은 최근 검출 위치
        self.last_good_lpos: Optional[int] = None
        self.last_good_rpos: Optional[int] = None
        self.last_good_center: Optional[int] = None
        self.last_good_lane_width: Optional[int] = None
        self.last_good_left_slope: Optional[float] = None
        self.last_good_right_slope: Optional[float] = None

    @staticmethod
    def _slope(line: Tuple[int, int, int, int]) -> Optional[float]:
        x1, y1, x2, y2 = line
        if x2 == x1:
            return None
        return float(y2 - y1) / float(x2 - x1)

    @staticmethod
    def _mid_x(line: Tuple[int, int, int, int]) -> float:
        x1, _, x2, _ = line
        return (x1 + x2) / 2.0

    def divide_left_right(
        self,
        lines: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        left: List[Tuple[int, int, int, int]] = []
        right: List[Tuple[int, int, int, int]] = []

        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            if x2 == x1:
                continue

            slope = float(y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.1 or abs(slope) > 10.0:
                continue

            if slope < 0 and x2 < self.cfg.width / 2 - self.cfg.center_margin:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > self.cfg.width / 2 + self.cfg.center_margin:
                right.append((x1, y1, x2, y2))

        return left, right

    def get_pos(self, lines: List[Tuple[int, int, int, int]]) -> Optional[int]:
        if len(lines) == 0:
            return None

        x_sum = 0.0
        y_sum = 0.0
        m_sum = 0.0
        valid_count = 0

        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            m = float(y2 - y1) / float(x2 - x1)
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += m
            valid_count += 1

        if valid_count == 0:
            return None

        x_avg = x_sum / (valid_count * 2.0)
        y_avg = y_sum / (valid_count * 2.0)
        m_avg = m_sum / valid_count

        if abs(m_avg) < 1e-12:
            return None

        b = y_avg - m_avg * x_avg
        x_at_roi_mid = int((self.cfg.gap / 2.0 - b) / m_avg)
        return x_at_roi_mid

    def _empty_result_base(self, gray: np.ndarray, edge: np.ndarray, roi_gray: np.ndarray, roi_edge: np.ndarray) -> Dict[str, Any]:
        return {
            "gray": gray,
            "edge": edge,
            "roi_gray": roi_gray,
            "roi_edge": roi_edge,
            "all_lines": [],
            "selected_left": [],
            "selected_right": [],
            "left_candidates": [],
            "right_candidates": [],
            "lpos": None,
            "rpos": None,
            "lane_width": None,
            "center": self.prev_center,
            "center_error": int(self.prev_center - self.cfg.width // 2),
            "mode": "NONE",
            "baseline_mode": "NONE",
            "corner": "STRAIGHT",
            "left_slope": None,
            "right_slope": None,
            "hough_line_count": 0,
            "left_line_count": 0,
            "right_line_count": 0,
            "roi_mean": float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0,
            "roi_std": float(np.std(roi_gray)) if roi_gray.size > 0 else 0.0,
            "roi_min": int(np.min(roi_gray)) if roi_gray.size > 0 else 0,
            "roi_max": int(np.max(roi_gray)) if roi_gray.size > 0 else 0,
            "edge_pixel_count": int(np.count_nonzero(roi_edge)) if roi_edge.size > 0 else 0,
            "fail_count": self.fail_count,
            "center_jump": 0,
            "lane_width_jump": None,
            "suspicious_reasons": "",
            "recovery_used": 0,
            "recovery_success": 0,
            "recovery_reason": "",
            "recovery_left_found": 0,
            "recovery_right_found": 0,
            "recovery_window_px": self.cfg.recovery_window_px,
            "local_first_used": 0,
            "local_accept": 0,
            "full_fallback_used": 0,
            "local_reason": "",
        }

    def _apply_corner_shift(self, center: int, left_slope: Optional[float]) -> Tuple[int, str]:
        corner = "STRAIGHT"
        if left_slope is not None:
            dev = left_slope - self.cfg.corner_left_base
            if dev < -self.cfg.corner_slope_thresh:
                corner = "LEFT"
        if corner == "LEFT":
            center -= self.cfg.corner_shift_px
        return center, corner

    def _finalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        center = int(max(0, min(cfg.width - 1, int(result["center"]))))
        result["center"] = center
        result["center_error"] = int(center - cfg.width // 2)

        if result["mode"] == "NONE":
            self.fail_count += 1
        else:
            self.fail_count = 0

        result["fail_count"] = self.fail_count

        center_jump = int(abs(center - self.prev_center))
        result["center_jump"] = center_jump

        lane_width_jump: Optional[int] = None
        if result["lane_width"] is not None and self.prev_lane_width is not None:
            lane_width_jump = int(abs(int(result["lane_width"]) - self.prev_lane_width))
        result["lane_width_jump"] = lane_width_jump

        suspicious_reasons: List[str] = []
        if result["mode"] == "NONE":
            suspicious_reasons.append("mode_NONE")
        if result["hough_line_count"] == 0:
            suspicious_reasons.append("hough_0")
        if center_jump > cfg.center_jump_threshold:
            suspicious_reasons.append("center_jump")
        if lane_width_jump is not None and lane_width_jump > cfg.lane_width_jump_threshold:
            suspicious_reasons.append("lane_width_jump")
        if int(result.get("recovery_used", 0)) == 1:
            suspicious_reasons.append("v0_2_local_first_used")
        if int(result.get("recovery_success", 0)) == 1:
            suspicious_reasons.append("v0_2_local_accept")
        result["suspicious_reasons"] = ";".join(suspicious_reasons)

        self.prev_center = center
        if result["lane_width"] is not None:
            self.prev_lane_width = float(result["lane_width"])

        # last_good은 양쪽 차선이 동시에 검출된 경우에만 강하게 갱신한다.
        if result["mode"] == "BOTH" and result["lpos"] is not None and result["rpos"] is not None:
            self.last_good_lpos = int(result["lpos"])
            self.last_good_rpos = int(result["rpos"])
            self.last_good_center = int(result["center"])
            self.last_good_lane_width = int(result["rpos"] - result["lpos"])
            self.last_good_left_slope = result["left_slope"]
            self.last_good_right_slope = result["right_slope"]

        return result

    def _run_full_roi_detection(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        cfg = self.cfg

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)

        y0 = max(0, int(cfg.offset))
        y1 = min(edge.shape[0], int(cfg.offset + cfg.gap))
        x0 = 0
        x1 = min(edge.shape[1], int(cfg.width))

        roi_gray = gray[y0:y1, x0:x1]
        roi_edge = edge[y0:y1, x0:x1]

        result = self._empty_result_base(gray, edge, roi_gray, roi_edge)

        lines = cv2.HoughLinesP(
            roi_edge,
            1,
            math.pi / 180.0,
            cfg.hough_threshold,
            minLineLength=cfg.hough_min_len,
            maxLineGap=cfg.hough_max_gap,
        )

        result["hough_line_count"] = 0 if lines is None else int(len(lines))

        if lines is None:
            result["mode"] = "NONE"
            result["baseline_mode"] = "NONE"
            result["center"] = self.prev_center
            return result

        all_lines = [tuple(map(int, line[0])) for line in lines]
        left_candidates, right_candidates = self.divide_left_right(lines)

        selected_left: List[Tuple[int, int, int, int]] = []
        selected_right: List[Tuple[int, int, int, int]] = []
        lpos: Optional[int] = None
        rpos: Optional[int] = None
        lane_width: Optional[int] = None
        left_slope: Optional[float] = None
        right_slope: Optional[float] = None

        if len(left_candidates) > 0:
            selected_left = [max(left_candidates, key=self._mid_x)]
            lpos = self.get_pos(selected_left)
            left_slope = self._slope(selected_left[0])

        if len(right_candidates) > 0:
            selected_right = [min(right_candidates, key=self._mid_x)]
            rpos = self.get_pos(selected_right)
            right_slope = self._slope(selected_right[0])

        if lpos is not None and rpos is not None:
            mode = "BOTH"
            center = (lpos + rpos) // 2
            lane_width = int(rpos - lpos)
            half = lane_width / 2.0
            if self.ema_half_width is None:
                self.ema_half_width = half
            else:
                alpha = cfg.ema_alpha
                self.ema_half_width = alpha * half + (1.0 - alpha) * self.ema_half_width
        elif lpos is not None:
            mode = "LEFT"
            if self.ema_half_width is not None:
                center = int(lpos + self.ema_half_width * cfg.one_lane_ratio)
            else:
                center = self.prev_center
        elif rpos is not None:
            mode = "RIGHT"
            if self.ema_half_width is not None:
                center = int(rpos - self.ema_half_width * cfg.one_lane_ratio)
            else:
                center = self.prev_center
        else:
            mode = "NONE"
            center = self.prev_center

        center, corner = self._apply_corner_shift(center, left_slope)

        result.update({
            "all_lines": all_lines,
            "selected_left": selected_left,
            "selected_right": selected_right,
            "left_candidates": left_candidates,
            "right_candidates": right_candidates,
            "lpos": lpos,
            "rpos": rpos,
            "lane_width": lane_width,
            "center": center,
            "mode": mode,
            "baseline_mode": mode,
            "corner": corner,
            "left_slope": left_slope,
            "right_slope": right_slope,
            "left_line_count": len(left_candidates),
            "right_line_count": len(right_candidates),
        })
        return result

    def _adaptive_canny(self, gray_roi: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        med = float(np.median(blur)) if blur.size > 0 else 0.0
        low = int(max(cfg.recovery_canny_low_min, (1.0 - cfg.recovery_canny_sigma) * med))
        high = int(min(cfg.recovery_canny_high_max, max(cfg.recovery_canny_high_min, (1.0 + cfg.recovery_canny_sigma) * med)))
        if high <= low:
            high = low + 30
        return cv2.Canny(blur, low, high)

    def _detect_local_side(
        self,
        roi_gray: np.ndarray,
        side: str,
        center_x: int,
        target_pos: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], Optional[int], Optional[float], np.ndarray]:
        cfg = self.cfg
        h, w = roi_gray.shape[:2]
        win = int(cfg.recovery_window_px)
        x0 = max(0, int(center_x - win))
        x1 = min(w, int(center_x + win))
        if x1 <= x0 + 5:
            return [], [], None, None, np.zeros_like(roi_gray)

        local_gray = roi_gray[:, x0:x1]
        # Version 0.2: local ROI만 적용한다. CLAHE/adaptive Canny는 사용하지 않는다.
        # Canny threshold는 version 0 baseline과 동일한 고정값을 사용한다.
        local_blur = cv2.GaussianBlur(local_gray, (5, 5), 0)
        local_edge = cv2.Canny(local_blur, cfg.canny_low, cfg.canny_high)

        local_lines = cv2.HoughLinesP(
            local_edge,
            1,
            math.pi / 180.0,
            max(12, int(cfg.hough_threshold * 0.6)),
            minLineLength=max(10, int(cfg.hough_min_len * 0.6)),
            maxLineGap=max(10, int(cfg.hough_max_gap * 1.5)),
        )

        edge_canvas = np.zeros_like(roi_gray)
        edge_canvas[:, x0:x1] = local_edge

        if local_lines is None:
            return [], [], None, None, edge_canvas

        all_lines: List[Tuple[int, int, int, int]] = []
        valid_lines: List[Tuple[int, int, int, int]] = []

        for line in local_lines:
            lx1, y1, lx2, y2 = map(int, line[0])
            gx1 = lx1 + x0
            gx2 = lx2 + x0
            global_line = (gx1, y1, gx2, y2)
            all_lines.append(global_line)
            slope = self._slope(global_line)
            if slope is None:
                continue
            if abs(slope) < 0.1 or abs(slope) > 10.0:
                continue
            if side == "left" and slope >= 0:
                continue
            if side == "right" and slope <= 0:
                continue
            pos = self.get_pos([global_line])
            if pos is None:
                continue
            if abs(pos - target_pos) > cfg.recovery_pos_gate_px:
                continue
            valid_lines.append(global_line)

        if not valid_lines:
            return all_lines, [], None, None, edge_canvas

        # last_good 위치와 가장 가까운 선분을 선택한다.
        def score(line: Tuple[int, int, int, int]) -> float:
            pos = self.get_pos([line])
            if pos is None:
                return 1e9
            return abs(pos - target_pos)

        selected = [min(valid_lines, key=score)]
        selected_pos = self.get_pos(selected)
        selected_slope = self._slope(selected[0])
        return all_lines, selected, selected_pos, selected_slope, edge_canvas

    def _attempt_recovery(self, full_result: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        result = dict(full_result)
        result["recovery_used"] = 1
        result["recovery_success"] = 0
        result["recovery_reason"] = "triggered_local_first"
        result["local_first_used"] = 1
        result["local_accept"] = 0
        result["full_fallback_used"] = 0
        result["local_reason"] = "triggered_local_first"

        if self.last_good_lpos is None or self.last_good_rpos is None or self.last_good_lane_width is None:
            result["recovery_reason"] = "no_last_good"
            return result

        roi_gray = full_result["roi_gray"]
        if roi_gray.size == 0:
            result["recovery_reason"] = "empty_roi"
            return result

        left_all, left_sel, lpos, left_slope, left_edge = self._detect_local_side(
            roi_gray, "left", self.last_good_lpos, self.last_good_lpos
        )
        right_all, right_sel, rpos, right_slope, right_edge = self._detect_local_side(
            roi_gray, "right", self.last_good_rpos, self.last_good_rpos
        )

        local_edge_union = cv2.bitwise_or(left_edge, right_edge)
        all_lines = left_all + right_all
        selected_left = left_sel
        selected_right = right_sel

        result["recovery_left_found"] = 1 if lpos is not None else 0
        result["recovery_right_found"] = 1 if rpos is not None else 0

        if lpos is not None and rpos is not None:
            lane_width = int(rpos - lpos)
            if lane_width <= 0:
                result["recovery_reason"] = "invalid_width"
                return result
            if abs(lane_width - self.last_good_lane_width) > cfg.recovery_lane_width_gate_px:
                result["recovery_reason"] = "lane_width_gate_fail"
                return result
            mode = "BOTH"
            center = (lpos + rpos) // 2
            half = lane_width / 2.0
            if self.ema_half_width is None:
                self.ema_half_width = half
            else:
                alpha = cfg.ema_alpha
                self.ema_half_width = alpha * half + (1.0 - alpha) * self.ema_half_width
        elif lpos is not None:
            mode = "LEFT"
            lane_width = None
            if self.ema_half_width is not None:
                center = int(lpos + self.ema_half_width * cfg.one_lane_ratio)
            elif self.last_good_lane_width is not None:
                center = int(lpos + self.last_good_lane_width / 2.0)
            else:
                result["recovery_reason"] = "left_only_no_width"
                return result
        elif rpos is not None:
            mode = "RIGHT"
            lane_width = None
            if self.ema_half_width is not None:
                center = int(rpos - self.ema_half_width * cfg.one_lane_ratio)
            elif self.last_good_lane_width is not None:
                center = int(rpos - self.last_good_lane_width / 2.0)
            else:
                result["recovery_reason"] = "right_only_no_width"
                return result
        else:
            result["recovery_reason"] = "no_local_line"
            return result

        center, corner = self._apply_corner_shift(center, left_slope)

        result.update({
            "roi_edge": local_edge_union,
            "all_lines": all_lines,
            "selected_left": selected_left,
            "selected_right": selected_right,
            "left_candidates": left_all,
            "right_candidates": right_all,
            "lpos": lpos,
            "rpos": rpos,
            "lane_width": lane_width,
            "center": center,
            "mode": mode,
            "corner": corner,
            "left_slope": left_slope,
            "right_slope": right_slope,
            "hough_line_count": len(all_lines),
            "left_line_count": len(left_all),
            "right_line_count": len(right_all),
            "edge_pixel_count": int(np.count_nonzero(local_edge_union)),
            "recovery_success": 1,
            "recovery_reason": "success",
            "local_accept": 1,
            "local_reason": "success",
        })
        return result

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        # 1) baseline full ROI 결과는 매 프레임 계산해 baseline_mode로 기록한다.
        #    이 값은 v0/baseline과 비교하기 위한 기준이다.
        full_result = self._run_full_roi_detection(frame_bgr)
        baseline_mode = full_result["mode"]

        # 2) Version 0.2: last_good 기반 local ROI를 매 프레임 우선 탐색한다.
        can_try_local = (
            self.last_good_lpos is not None
            and self.last_good_rpos is not None
            and self.last_good_lane_width is not None
        )

        if can_try_local:
            local_result = self._attempt_recovery(full_result)
            local_result["baseline_mode"] = baseline_mode
            local_result["local_first_used"] = 1
            local_result["local_accept"] = int(local_result.get("recovery_success", 0) == 1)
            local_result["local_reason"] = local_result.get("recovery_reason", "")
            local_result["full_fallback_used"] = 0

            # local ROI 결과가 검증을 통과하면 최종 결과로 채택한다.
            if int(local_result.get("local_accept", 0)) == 1:
                return self._finalize_result(local_result)

            # local ROI가 실패하면 full ROI 결과로 fallback한다.
            full_result["baseline_mode"] = baseline_mode
            full_result["local_first_used"] = 1
            full_result["local_accept"] = 0
            full_result["full_fallback_used"] = 1
            full_result["local_reason"] = local_result.get("recovery_reason", "local_fail")
            full_result["recovery_used"] = 1
            full_result["recovery_success"] = 0
            full_result["recovery_reason"] = "fallback_after_local_fail:{}".format(
                local_result.get("recovery_reason", "local_fail")
            )
            return self._finalize_result(full_result)

        # 3) last_good이 아직 없으면 local-first를 수행할 수 없으므로 full ROI 결과를 사용한다.
        full_result["baseline_mode"] = baseline_mode
        full_result["local_first_used"] = 0
        full_result["local_accept"] = 0
        full_result["full_fallback_used"] = 1
        full_result["local_reason"] = "no_last_good"
        full_result["recovery_used"] = 0
        full_result["recovery_success"] = 0
        full_result["recovery_reason"] = "no_last_good"
        return self._finalize_result(full_result)

@dataclass
class SegmentRuntime:
    name: str
    start_sec: float
    end_sec: float
    detector: BaselineLaneDetector
    csv_file: Any
    writer: csv.DictWriter
    processed_count: int = 0
    mode_count: Optional[Dict[str, int]] = None
    entered: bool = False
    finished: bool = False

    def __post_init__(self) -> None:
        if self.mode_count is None:
            self.mode_count = {"BOTH": 0, "LEFT": 0, "RIGHT": 0, "NONE": 0}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", name.strip())
    return cleaned if cleaned else "segment"


def validate_segments(segments: List[Dict[str, Any]]) -> None:
    if not segments:
        raise ValueError("TIME_SEGMENTS가 비어 있습니다.")

    for idx, seg in enumerate(segments):
        if "name" not in seg or "start_sec" not in seg or "end_sec" not in seg:
            raise ValueError("TIME_SEGMENTS[{}]에는 name, start_sec, end_sec가 필요합니다.".format(idx))
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        if end <= start:
            raise ValueError("TIME_SEGMENTS[{}] end_sec는 start_sec보다 커야 합니다.".format(idx))


def to_bgr_from_msg(bridge: CvBridge, msg: Any) -> np.ndarray:
    """
    rosbag에서 읽은 이미지 메시지를 OpenCV BGR 이미지로 변환한다.
    """
    msg_type = getattr(msg, "_type", "")

    if msg_type == "sensor_msgs/CompressedImage":
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("CompressedImage decode 실패")
        return image

    if msg_type == "sensor_msgs/Image":
        encoding = str(getattr(msg, "encoding", "")).lower()

        if encoding in ["bgr8", "rgb8", "mono8", "8uc1", "8uc3"]:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return img

        if encoding in ["yuv422", "yuyv", "yuy2"]:
            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if raw.size < h * step:
                raise RuntimeError("YUV422 data size 부족: raw={} expected={}".format(raw.size, h * step))
            raw2d = raw[:h * step].reshape((h, step))
            yuyv = raw2d[:, :w * 2].reshape((h, w, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

        if encoding in ["uyvy"]:
            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if raw.size < h * step:
                raise RuntimeError("UYVY data size 부족: raw={} expected={}".format(raw.size, h * step))
            raw2d = raw[:h * step].reshape((h, step))
            uyvy = raw2d[:, :w * 2].reshape((h, w, 2))
            return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            if encoding == "rgb8":
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        raise RuntimeError("지원하지 않는 Image encoding/shape: encoding={} shape={}".format(encoding, getattr(img, "shape", None)))

    raise TypeError("지원하지 않는 메시지 타입: msg._type={} python_type={}".format(msg_type, type(msg)))


def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.55, thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, result: Dict[str, Any], cfg: BaselineConfig, segment_name: str, rel_time: float) -> np.ndarray:
    overlay = frame_bgr.copy()

    h, w = overlay.shape[:2]
    y0 = max(0, int(cfg.offset))
    y1 = min(h, int(cfg.offset + cfg.gap))

    cv2.rectangle(overlay, (0, y0), (min(cfg.width, w) - 1, y1 - 1), (255, 0, 0), 2)
    cv2.line(overlay, (cfg.width // 2, 0), (cfg.width // 2, h - 1), (120, 120, 120), 1)

    for x1, y1_l, x2, y2_l in result["all_lines"]:
        cv2.line(overlay, (x1, y1_l + y0), (x2, y2_l + y0), (0, 255, 255), 1)

    for x1, y1_l, x2, y2_l in result["selected_left"]:
        cv2.line(overlay, (x1, y1_l + y0), (x2, y2_l + y0), (0, 255, 0), 3)
    for x1, y1_l, x2, y2_l in result["selected_right"]:
        cv2.line(overlay, (x1, y1_l + y0), (x2, y2_l + y0), (255, 0, 255), 3)

    roi_mid_y = int(cfg.offset + cfg.gap / 2)

    if result["lpos"] is not None:
        cv2.circle(overlay, (int(result["lpos"]), roi_mid_y), 6, (0, 255, 0), -1)
        draw_text(overlay, "L", (int(result["lpos"]) + 8, roi_mid_y - 8))

    if result["rpos"] is not None:
        cv2.circle(overlay, (int(result["rpos"]), roi_mid_y), 6, (255, 0, 255), -1)
        draw_text(overlay, "R", (int(result["rpos"]) + 8, roi_mid_y - 8))

    center = int(result["center"])
    cv2.line(overlay, (center, y0), (center, y1), (0, 0, 255), 2)
    cv2.circle(overlay, (center, roi_mid_y), 7, (0, 0, 255), -1)

    draw_text(overlay, "segment={}  rel_t={:.3f}s".format(segment_name, rel_time), (10, 25))
    draw_text(overlay, "mode={}  center={}  err={}  fail={}".format(
        result["mode"], result["center"], result["center_error"], result["fail_count"]), (10, 50))
    draw_text(overlay, "roi_mean={:.1f}  edge={}  hough={}  L/R={}/{}".format(
        result["roi_mean"], result["edge_pixel_count"], result["hough_line_count"],
        result["left_line_count"], result["right_line_count"]), (10, 75))
    draw_text(overlay, "lpos={}  rpos={}  width={}  corner={}".format(
        result["lpos"], result["rpos"], result["lane_width"], result["corner"]), (10, 100))

    draw_text(overlay, "baseline={}  local_used={}  local_ok={}  fallback={}  reason={}".format(
        result.get("baseline_mode", ""), result.get("local_first_used", 0),
        result.get("local_accept", 0), result.get("full_fallback_used", 0),
        result.get("local_reason", result.get("recovery_reason", ""))), (10, 125))

    if result["suspicious_reasons"]:
        draw_text(overlay, "WARN: {}".format(result["suspicious_reasons"]), (10, 150))

    return overlay


def make_panel(img: np.ndarray, title: str, size: Tuple[int, int]) -> np.ndarray:
    if img.ndim == 2:
        panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        panel = img.copy()
    panel = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (size[0] - 1, 22), (0, 0, 0), -1)
    cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def make_debug_grid(frame_bgr: np.ndarray, overlay: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    panel_w, panel_h = 320, 240

    roi_hough = cv2.cvtColor(result["roi_edge"], cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in result["all_lines"]:
        cv2.line(roi_hough, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for x1, y1, x2, y2 in result["selected_left"]:
        cv2.line(roi_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x1, y1, x2, y2 in result["selected_right"]:
        cv2.line(roi_hough, (x1, y1), (x2, y2), (255, 0, 255), 2)

    p1 = make_panel(frame_bgr, "Original", (panel_w, panel_h))
    p2 = make_panel(result["gray"], "Gray", (panel_w, panel_h))
    p3 = make_panel(result["edge"], "Canny edge", (panel_w, panel_h))
    p4 = make_panel(result["roi_gray"], "ROI gray", (panel_w, panel_h))
    p5 = make_panel(roi_hough, "ROI Hough", (panel_w, panel_h))
    p6 = make_panel(overlay, "Final overlay", (panel_w, panel_h))

    top = np.hstack([p1, p2, p3])
    bottom = np.hstack([p4, p5, p6])
    return np.vstack([top, bottom])


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def csv_fieldnames() -> List[str]:
    return [
        "segment_name",
        "segment_start_sec",
        "segment_end_sec",
        "frame_idx",
        "processed_idx",
        "bag_abs_time",
        "bag_rel_time",
        "roi_mean",
        "roi_std",
        "roi_min",
        "roi_max",
        "edge_pixel_count",
        "hough_line_count",
        "left_line_count",
        "right_line_count",
        "left_slope",
        "right_slope",
        "lpos",
        "rpos",
        "lane_width",
        "center",
        "center_error",
        "mode",
        "baseline_mode",
        "corner",
        "fail_count",
        "center_jump",
        "lane_width_jump",
        "recovery_used",
        "recovery_success",
        "recovery_left_found",
        "recovery_right_found",
        "recovery_window_px",
        "recovery_reason",
        "local_first_used",
        "local_accept",
        "full_fallback_used",
        "local_reason",
        "suspicious_reasons",
    ]


def open_segment_csv(output_dir: str, segment_name: str) -> Tuple[Any, csv.DictWriter, str]:
    filename = safe_filename(segment_name) + ".csv"
    csv_path = os.path.join(output_dir, filename)
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
    writer.writeheader()
    return f, writer, csv_path



def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def summarize_jump_stability(csv_path: str, center_threshold: int, lane_width_threshold: int) -> Dict[str, Any]:
    """
    저장된 CSV를 다시 읽어서 center_jump와 lane_width_jump 안정성 지표를 계산한다.
    알고리즘 결과에는 영향을 주지 않고, 종료 요약 출력에만 사용한다.
    """
    total_rows = 0
    center_values: List[float] = []
    lane_width_values: List[float] = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1

                cj = _safe_float(row.get("center_jump"))
                if cj is not None:
                    center_values.append(cj)

                lwj = _safe_float(row.get("lane_width_jump"))
                if lwj is not None:
                    lane_width_values.append(lwj)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "total_rows": total_rows,
        }

    center_bad = sum(1 for v in center_values if v >= center_threshold)
    lane_width_bad = sum(1 for v in lane_width_values if v >= lane_width_threshold)

    center_count = len(center_values)
    lane_width_count = len(lane_width_values)

    return {
        "ok": True,
        "total_rows": total_rows,
        "center_count": center_count,
        "center_max": None if center_count == 0 else int(max(center_values)),
        "center_bad": center_bad,
        "center_bad_pct": 0.0 if center_count == 0 else 100.0 * center_bad / center_count,
        "center_threshold": center_threshold,
        "lane_width_count": lane_width_count,
        "lane_width_max": None if lane_width_count == 0 else int(max(lane_width_values)),
        "lane_width_bad": lane_width_bad,
        "lane_width_bad_pct": 0.0 if lane_width_count == 0 else 100.0 * lane_width_bad / lane_width_count,
        "lane_width_threshold": lane_width_threshold,
    }


def print_jump_stability_summary(csv_path: str, center_threshold: int, lane_width_threshold: int) -> None:
    summary = summarize_jump_stability(csv_path, center_threshold, lane_width_threshold)
    if not summary.get("ok", False):
        print("  jump 안정성      : 계산 실패 | {}".format(summary.get("error", "unknown error")))
        return

    center_max = "NA" if summary["center_max"] is None else summary["center_max"]
    lane_width_max = "NA" if summary["lane_width_max"] is None else summary["lane_width_max"]

    print(
        "  jump 안정성      : center_max={} | center_jump>={}px: {}/{} ({:.2f}%) | "
        "lane_width_max={} | lane_width_jump>={}px: {}/{} ({:.2f}%)".format(
            center_max,
            summary["center_threshold"],
            summary["center_bad"],
            summary["center_count"],
            summary["center_bad_pct"],
            lane_width_max,
            summary["lane_width_threshold"],
            summary["lane_width_bad"],
            summary["lane_width_count"],
            summary["lane_width_bad_pct"],
        )
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rosbag 한 개에서 코드 내부 TIME_SEGMENTS 구간만 버전 0.2 local ROI-only 차선 검출 분석 + 구간별 CSV 저장"
    )

    parser.add_argument("--bag", required=True, help="입력 rosbag 파일 경로")
    parser.add_argument("--topic", default="/usb_cam/image_raw", help="카메라 이미지 토픽")
    parser.add_argument("--output-dir", required=True, help="구간별 CSV 저장 폴더")

    parser.add_argument("--view", choices=["overlay", "debug", "both", "none"], default="overlay",
                        help="실행 중 화면 시각화 방식. mp4/png 저장은 하지 않음")
    parser.add_argument("--display-scale", type=float, default=1.0, help="표시 창 크기 배율")
    parser.add_argument("--delay-ms", type=int, default=80, help="cv2.waitKey 대기 시간. 값이 클수록 느리게 재생됨")
    parser.add_argument("--every-n", type=int, default=1, help="N프레임마다 1개 처리")
    parser.add_argument("--max-frames-per-segment", type=int, default=0, help="구간별 최대 처리 프레임 수. 0이면 제한 없음")

    # 기존 baseline 파라미터. 기본값은 현재 코드 기준.
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--canny-low", type=int, default=40)
    parser.add_argument("--canny-high", type=int, default=100)
    parser.add_argument("--offset", type=int, default=330)
    parser.add_argument("--gap", type=int, default=110)
    parser.add_argument("--hough-threshold", type=int, default=30)
    parser.add_argument("--hough-min-len", type=int, default=20)
    parser.add_argument("--hough-max-gap", type=int, default=10)
    parser.add_argument("--center-margin", type=int, default=90)

    # Version 0.2 local ROI parameters
    parser.add_argument("--recovery-fail-threshold", type=int, default=3,
                        help="버전 2에서는 호환성용 인자. local-first는 last_good이 있으면 매 프레임 시도")
    parser.add_argument("--recovery-window-px", type=int, default=90,
                        help="last_good_lpos/rpos 주변 local ROI 반폭 [px]")
    parser.add_argument("--recovery-pos-gate-px", type=int, default=100,
                        help="복구 후보 위치가 last_good과 허용되는 최대 거리 [px]")
    parser.add_argument("--recovery-lane-width-gate-px", type=int, default=130,
                        help="복구된 lane_width와 last_good_lane_width의 허용 차이 [px]")

    return parser.parse_args()


def find_active_segments(segments: List[SegmentRuntime], rel_time: float) -> List[SegmentRuntime]:
    active: List[SegmentRuntime] = []
    for seg in segments:
        if seg.finished:
            continue
        if rel_time > seg.end_sec:
            if seg.entered:
                seg.finished = True
            continue
        if seg.start_sec <= rel_time <= seg.end_sec:
            active.append(seg)
    return active


def handle_view(
    view: str,
    overlay: np.ndarray,
    frame_bgr: np.ndarray,
    result: Dict[str, Any],
    display_scale: float,
    delay_ms: int,
    paused: bool,
) -> Tuple[bool, bool]:
    """
    반환: (continue_running, paused)
    """
    if view == "none":
        return True, paused

    if view in ["overlay", "both"]:
        cv2.imshow("version0.2 section overlay - no save", resize_for_display(overlay, display_scale))

    if view in ["debug", "both"]:
        debug_grid = make_debug_grid(frame_bgr, overlay, result)
        cv2.imshow("version0.2 section debug - no save", resize_for_display(debug_grid, display_scale))

    while True:
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in [ord("q"), 27]:
            print("[INFO] 사용자 종료")
            return False, paused

        if key == ord(" "):
            paused = not paused
            print("[INFO] paused={}".format(paused))

        if paused:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 in [ord("q"), 27]:
                print("[INFO] 사용자 종료")
                return False, paused
            if key2 == ord(" "):
                paused = False
                print("[INFO] paused=False")
                break
            if key2 == ord("n"):
                return True, True
            continue

        break

    return True, paused


def main() -> None:
    args = parse_args()

    try:
        validate_segments(TIME_SEGMENTS)
    except Exception as exc:
        print("[ERROR] TIME_SEGMENTS 설정 오류: {}".format(exc))
        sys.exit(1)

    if not os.path.isfile(args.bag):
        print("[ERROR] bag 파일이 없습니다: {}".format(args.bag))
        sys.exit(1)

    if args.every_n < 1:
        print("[ERROR] --every-n은 1 이상이어야 합니다.")
        sys.exit(1)

    ensure_dir(args.output_dir)

    cfg = BaselineConfig(
        width=args.width,
        height=args.height,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        offset=args.offset,
        gap=args.gap,
        hough_threshold=args.hough_threshold,
        hough_min_len=args.hough_min_len,
        hough_max_gap=args.hough_max_gap,
        center_margin=args.center_margin,
        recovery_fail_threshold=args.recovery_fail_threshold,
        recovery_window_px=args.recovery_window_px,
        recovery_pos_gate_px=args.recovery_pos_gate_px,
        recovery_lane_width_gate_px=args.recovery_lane_width_gate_px,
    )

    bridge = CvBridge()
    csv_paths: Dict[str, str] = {}
    segments: List[SegmentRuntime] = []

    for seg_cfg in TIME_SEGMENTS:
        name = str(seg_cfg["name"])
        csv_file, writer, csv_path = open_segment_csv(args.output_dir, name)
        csv_paths[name] = csv_path
        segments.append(
            SegmentRuntime(
                name=name,
                start_sec=float(seg_cfg["start_sec"]),
                end_sec=float(seg_cfg["end_sec"]),
                detector=BaselineLaneDetector(cfg),
                csv_file=csv_file,
                writer=writer,
            )
        )

    print("[INFO] 입력 bag     : {}".format(args.bag))
    print("[INFO] 이미지 토픽  : {}".format(args.topic))
    print("[INFO] 출력 폴더    : {}".format(args.output_dir))
    print("[INFO] 화면 표시    : {}".format(args.view))
    print("[INFO] mp4/png 저장 : 하지 않음")
    print("[INFO] 시간 기준    : 이미지 토픽 첫 메시지 기준 상대시간 [s]")
    print("[INFO] 분석 구간:")
    for seg in segments:
        print("  - {}: {:.3f}s ~ {:.3f}s -> {}".format(seg.name, seg.start_sec, seg.end_sec, csv_paths[seg.name]))

    total_msg_count = 0
    first_topic_time: Optional[float] = None
    decode_fail_count = 0
    paused = False
    continue_running = True

    try:
        with rosbag.Bag(args.bag, "r") as bag:
            topic_infos = bag.get_type_and_topic_info().topics
            if args.topic not in topic_infos:
                print("[ERROR] bag 안에 지정한 토픽이 없습니다: {}".format(args.topic))
                print("[INFO] bag 내 토픽 목록:")
                for topic_name in sorted(topic_infos.keys()):
                    print("  - {}".format(topic_name))
                sys.exit(1)

            selected_info = topic_infos[args.topic]
            print("[INFO] bag topic type : {}".format(selected_info.msg_type))

            for topic, msg, stamp in bag.read_messages(topics=[args.topic]):
                abs_time = stamp.to_sec()
                if first_topic_time is None:
                    first_topic_time = abs_time
                rel_time = abs_time - first_topic_time

                if total_msg_count % args.every_n != 0:
                    total_msg_count += 1
                    continue

                active_segments = find_active_segments(segments, rel_time)
                if not active_segments:
                    total_msg_count += 1
                    # 모든 구간이 끝났으면 더 볼 필요 없음
                    if all(seg.finished or rel_time > seg.end_sec for seg in segments):
                        break
                    continue

                try:
                    frame_bgr = to_bgr_from_msg(bridge, msg)
                except Exception as exc:
                    decode_fail_count += 1
                    if decode_fail_count <= 10 or decode_fail_count % 100 == 0:
                        print("[WARN] frame decode 실패: frame_idx={} rel_time={:.3f} | msg_type={} | encoding={} | {}".format(
                            total_msg_count,
                            rel_time,
                            getattr(msg, "_type", ""),
                            getattr(msg, "encoding", ""),
                            exc,
                        ))
                    total_msg_count += 1
                    continue

                for seg in active_segments:
                    if args.max_frames_per_segment > 0 and seg.processed_count >= args.max_frames_per_segment:
                        seg.finished = True
                        continue

                    seg.entered = True
                    result = seg.detector.process(frame_bgr)
                    assert seg.mode_count is not None
                    seg.mode_count[result["mode"]] = seg.mode_count.get(result["mode"], 0) + 1

                    row = {
                        "segment_name": seg.name,
                        "segment_start_sec": "{:.6f}".format(seg.start_sec),
                        "segment_end_sec": "{:.6f}".format(seg.end_sec),
                        "frame_idx": total_msg_count,
                        "processed_idx": seg.processed_count,
                        "bag_abs_time": "{:.9f}".format(abs_time),
                        "bag_rel_time": "{:.9f}".format(rel_time),
                        "roi_mean": "{:.6f}".format(result["roi_mean"]),
                        "roi_std": "{:.6f}".format(result["roi_std"]),
                        "roi_min": result["roi_min"],
                        "roi_max": result["roi_max"],
                        "edge_pixel_count": result["edge_pixel_count"],
                        "hough_line_count": result["hough_line_count"],
                        "left_line_count": result["left_line_count"],
                        "right_line_count": result["right_line_count"],
                        "left_slope": "" if result["left_slope"] is None else "{:.9f}".format(result["left_slope"]),
                        "right_slope": "" if result["right_slope"] is None else "{:.9f}".format(result["right_slope"]),
                        "lpos": "" if result["lpos"] is None else result["lpos"],
                        "rpos": "" if result["rpos"] is None else result["rpos"],
                        "lane_width": "" if result["lane_width"] is None else result["lane_width"],
                        "center": result["center"],
                        "center_error": result["center_error"],
                        "mode": result["mode"],
                        "baseline_mode": result.get("baseline_mode", result["mode"]),
                        "corner": result["corner"],
                        "fail_count": result["fail_count"],
                        "center_jump": result["center_jump"],
                        "lane_width_jump": "" if result["lane_width_jump"] is None else result["lane_width_jump"],
                        "recovery_used": result.get("recovery_used", 0),
                        "recovery_success": result.get("recovery_success", 0),
                        "recovery_left_found": result.get("recovery_left_found", 0),
                        "recovery_right_found": result.get("recovery_right_found", 0),
                        "recovery_window_px": result.get("recovery_window_px", cfg.recovery_window_px),
                        "recovery_reason": result.get("recovery_reason", ""),
                        "local_first_used": result.get("local_first_used", 0),
                        "local_accept": result.get("local_accept", 0),
                        "full_fallback_used": result.get("full_fallback_used", 0),
                        "local_reason": result.get("local_reason", ""),
                        "suspicious_reasons": result["suspicious_reasons"],
                    }
                    seg.writer.writerow(row)

                    if seg.processed_count % 50 == 0:
                        seg.csv_file.flush()
                        print("[INFO] segment={} | processed={} | frame_idx={} | rel_t={:.3f}s | mode={} | baseline={} | local={}/{} | fallback={} | center={} | roi_mean={:.1f} | hough={}".format(
                            seg.name,
                            seg.processed_count,
                            total_msg_count,
                            rel_time,
                            result["mode"],
                            result.get("baseline_mode", result["mode"]),
                            result.get("local_first_used", 0),
                            result.get("local_accept", 0),
                            result.get("full_fallback_used", 0),
                            result["center"],
                            result["roi_mean"],
                            result["hough_line_count"],
                        ))

                    if args.view != "none":
                        overlay = draw_overlay(frame_bgr, result, cfg, seg.name, rel_time)
                        continue_running, paused = handle_view(
                            args.view,
                            overlay,
                            frame_bgr,
                            result,
                            args.display_scale,
                            args.delay_ms,
                            paused,
                        )
                        if not continue_running:
                            raise KeyboardInterrupt

                    seg.processed_count += 1

                total_msg_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        for seg in segments:
            seg.csv_file.flush()
            seg.csv_file.close()
        if args.view != "none":
            cv2.destroyAllWindows()

    print("\n========== 구간 분석 종료 ==========")
    print("전체 읽은 frame index 기준 마지막 값 : {}".format(total_msg_count))
    for seg in segments:
        print("\n[{}]".format(seg.name))
        print("  시간 구간        : {:.3f}s ~ {:.3f}s".format(seg.start_sec, seg.end_sec))
        print("  처리 프레임 수   : {}".format(seg.processed_count))
        print("  CSV 저장         : {}".format(csv_paths[seg.name]))
        print("  mode count       : {}".format(seg.mode_count))
        print_jump_stability_summary(
            csv_paths[seg.name],
            cfg.center_jump_threshold,
            cfg.lane_width_jump_threshold,
        )
    print("\nmp4/png 저장       : 없음")


if __name__ == "__main__":
    main()
