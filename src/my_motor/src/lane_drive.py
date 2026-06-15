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
    def __init__(self, cfg):
        self.cfg = cfg
        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()

        # 기존 main.py가 참조하는 값 유지
        self.ema_half_width = None
        self.prev_center = self.cfg.width // 2
        self.prev_lane_width = None
        self.fail_count = 0

        # v3_fixed local-first 상태값
        self.last_good_lpos = None
        self.last_good_rpos = None
        self.last_good_center = None
        self.last_good_lane_width = None
        self.last_good_left_slope = None
        self.last_good_right_slope = None

        # prediction mismatch 후보 추적용
        self.tentative_center = None
        self.tentative_lane_width = None
        self.tentative_lpos = None
        self.tentative_rpos = None
        self.tentative_count = 0

        # 디버그 오버레이용 최근 결과
        self.last_debug_result = None

        # config.py를 수정하지 않아도 동작하도록 여기서 ROS param fallback 처리
        self.center_jump_threshold = int(rospy.get_param("~center_jump_threshold", 80))
        self.lane_width_jump_threshold = int(rospy.get_param("~lane_width_jump_threshold", 100))

        self.recovery_window_px = int(rospy.get_param("~recovery_window_px", 90))
        self.recovery_pos_gate_px = int(rospy.get_param("~recovery_pos_gate_px", 100))
        self.recovery_lane_width_gate_px = int(rospy.get_param("~recovery_lane_width_gate_px", 130))
        self.recovery_clahe_clip = float(rospy.get_param("~recovery_clahe_clip", 2.0))
        self.recovery_canny_sigma = float(rospy.get_param("~recovery_canny_sigma", 0.33))
        self.recovery_canny_low_min = int(rospy.get_param("~recovery_canny_low_min", 15))
        self.recovery_canny_high_min = int(rospy.get_param("~recovery_canny_high_min", 45))
        self.recovery_canny_high_max = int(rospy.get_param("~recovery_canny_high_max", 180))

        self.v3_mismatch_center_gate_px = int(rospy.get_param("~v3_mismatch_center_gate_px", 70))
        self.v3_mismatch_lane_width_gate_px = int(rospy.get_param("~v3_mismatch_lane_width_gate_px", 90))
        self.v3_tentative_accept_frames = int(rospy.get_param("~v3_tentative_accept_frames", 2))
        self.v3_tentative_center_gate_px = int(rospy.get_param("~v3_tentative_center_gate_px", 50))
        self.v3_tentative_width_gate_px = int(rospy.get_param("~v3_tentative_width_gate_px", 80))
        self.v3_full_width_min_px = int(rospy.get_param("~v3_full_width_min_px", 120))
        self.v3_full_width_max_px = int(rospy.get_param("~v3_full_width_max_px", 520))

        rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback)

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    @staticmethod
    def _slope(line):
        x1, y1, x2, y2 = line
        if x2 == x1:
            return None
        return float(y2 - y1) / float(x2 - x1)

    @staticmethod
    def _mid_x(line):
        x1, y1, x2, y2 = line
        return (x1 + x2) / 2.0

    def divide_left_right(self, lines):
        left, right = [], []
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

    def get_pos(self, lines):
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
        return int((self.cfg.gap / 2.0 - b) / m_avg)

    def _empty_result_base(self, gray, edge, roi_gray, roi_edge):
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
            "fail_count": self.fail_count,
            "center_jump": 0,
            "lane_width_jump": None,
            "recovery_used": 0,
            "recovery_success": 0,
            "recovery_reason": "",
            "recovery_left_found": 0,
            "recovery_right_found": 0,
            "local_first_used": 0,
            "local_accept": 0,
            "full_fallback_used": 0,
            "local_reason": "",
            "v3_full_recheck_used": 0,
            "v3_mismatch": 0,
            "v3_tentative_count": 0,
            "v3_full_recheck_accept": 0,
            "v3_decision": "",
            "local_left_roi": None,
            "local_right_roi": None,
            "pred_lpos": self.last_good_lpos,
            "pred_rpos": self.last_good_rpos,
            "final_source": "FULL",
        }

    def _apply_corner_shift(self, center, left_slope):
        corner = "STRAIGHT"
        if left_slope is not None:
            dev = left_slope - self.cfg.corner_left_base
            if dev < -self.cfg.corner_slope_thresh:
                corner = "LEFT"
        if corner == "LEFT":
            center -= self.cfg.corner_shift_px
        return center, corner

    def _finalize_result(self, result):
        center = int(max(0, min(self.cfg.width - 1, int(result["center"]))))
        result["center"] = center
        result["center_error"] = int(center - self.cfg.width // 2)

        if result["mode"] == "NONE":
            self.fail_count += 1
        else:
            self.fail_count = 0
        result["fail_count"] = self.fail_count

        result["center_jump"] = int(abs(center - self.prev_center))

        lane_width_jump = None
        if result["lane_width"] is not None and self.prev_lane_width is not None:
            lane_width_jump = int(abs(int(result["lane_width"]) - self.prev_lane_width))
        result["lane_width_jump"] = lane_width_jump

        self.prev_center = center
        if result["lane_width"] is not None:
            self.prev_lane_width = float(result["lane_width"])

        # 양쪽 차선이 동시에 검출된 프레임만 last_good으로 사용
        if result["mode"] == "BOTH" and result["lpos"] is not None and result["rpos"] is not None:
            self.last_good_lpos = int(result["lpos"])
            self.last_good_rpos = int(result["rpos"])
            self.last_good_center = int(result["center"])
            self.last_good_lane_width = int(result["rpos"] - result["lpos"])
            self.last_good_left_slope = result["left_slope"]
            self.last_good_right_slope = result["right_slope"]

        self.last_debug_result = result
        return result

    def _run_full_roi_detection(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)

        y0 = max(0, int(self.cfg.offset))
        y1 = min(edge.shape[0], int(self.cfg.offset + self.cfg.gap))
        x0 = 0
        x1 = min(edge.shape[1], int(self.cfg.width))

        roi_gray = gray[y0:y1, x0:x1]
        roi_edge = edge[y0:y1, x0:x1]
        result = self._empty_result_base(gray, edge, roi_gray, roi_edge)

        lines = cv2.HoughLinesP(
            roi_edge,
            1,
            math.pi / 180.0,
            self.cfg.hough_threshold,
            minLineLength=self.cfg.hough_min_len,
            maxLineGap=self.cfg.hough_max_gap,
        )

        result["hough_line_count"] = 0 if lines is None else int(len(lines))
        if lines is None:
            result["mode"] = "NONE"
            result["baseline_mode"] = "NONE"
            result["center"] = self.prev_center
            return result

        all_lines = [tuple(map(int, line[0])) for line in lines]
        left_candidates, right_candidates = self.divide_left_right(lines)

        selected_left, selected_right = [], []
        lpos, rpos = None, None
        lane_width = None
        left_slope, right_slope = None, None

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
                alpha = self.cfg.ema_alpha
                self.ema_half_width = alpha * half + (1.0 - alpha) * self.ema_half_width
        elif lpos is not None:
            mode = "LEFT"
            center = int(lpos + self.ema_half_width * self.cfg.one_lane_ratio) if self.ema_half_width is not None else self.prev_center
        elif rpos is not None:
            mode = "RIGHT"
            center = int(rpos - self.ema_half_width * self.cfg.one_lane_ratio) if self.ema_half_width is not None else self.prev_center
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

    def _adaptive_canny(self, gray_roi):
        blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        med = float(np.median(blur)) if blur.size > 0 else 0.0
        low = int(max(self.recovery_canny_low_min, (1.0 - self.recovery_canny_sigma) * med))
        high = int(min(self.recovery_canny_high_max,
                       max(self.recovery_canny_high_min, (1.0 + self.recovery_canny_sigma) * med)))
        if high <= low:
            high = low + 30
        return cv2.Canny(blur, low, high)

    def _detect_local_side(self, roi_gray, side, center_x, target_pos):
        h, w = roi_gray.shape[:2]
        win = int(self.recovery_window_px)
        x0 = max(0, int(center_x - win))
        x1 = min(w, int(center_x + win))
        if x1 <= x0 + 5:
            return [], [], None, None, np.zeros_like(roi_gray), (x0, x1)

        local_gray = roi_gray[:, x0:x1]
        clahe = cv2.createCLAHE(clipLimit=self.recovery_clahe_clip, tileGridSize=(8, 8))
        local_eq = clahe.apply(local_gray)
        local_edge = self._adaptive_canny(local_eq)

        local_lines = cv2.HoughLinesP(
            local_edge,
            1,
            math.pi / 180.0,
            max(12, int(self.cfg.hough_threshold * 0.6)),
            minLineLength=max(10, int(self.cfg.hough_min_len * 0.6)),
            maxLineGap=max(10, int(self.cfg.hough_max_gap * 1.5)),
        )

        edge_canvas = np.zeros_like(roi_gray)
        edge_canvas[:, x0:x1] = local_edge

        if local_lines is None:
            return [], [], None, None, edge_canvas, (x0, x1)

        all_lines, valid_lines = [], []
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
            if abs(pos - target_pos) > self.recovery_pos_gate_px:
                continue
            valid_lines.append(global_line)

        if not valid_lines:
            return all_lines, [], None, None, edge_canvas, (x0, x1)

        def score(line):
            pos = self.get_pos([line])
            if pos is None:
                return 1e9
            return abs(pos - target_pos)

        selected = [min(valid_lines, key=score)]
        selected_pos = self.get_pos(selected)
        selected_slope = self._slope(selected[0])
        return all_lines, selected, selected_pos, selected_slope, edge_canvas, (x0, x1)

    def _attempt_local_first(self, full_result):
        result = dict(full_result)
        result["recovery_used"] = 1
        result["recovery_success"] = 0
        result["recovery_reason"] = "triggered_local_first"
        result["local_first_used"] = 1
        result["local_accept"] = 0
        result["full_fallback_used"] = 0
        result["local_reason"] = "triggered_local_first"
        result["pred_lpos"] = self.last_good_lpos
        result["pred_rpos"] = self.last_good_rpos
        result["final_source"] = "LOCAL_TRY"

        if self.last_good_lpos is None or self.last_good_rpos is None or self.last_good_lane_width is None:
            result["recovery_reason"] = "no_last_good"
            return result

        roi_gray = full_result["roi_gray"]
        if roi_gray.size == 0:
            result["recovery_reason"] = "empty_roi"
            return result

        left_all, left_sel, lpos, left_slope, left_edge, left_roi = self._detect_local_side(
            roi_gray, "left", self.last_good_lpos, self.last_good_lpos
        )
        right_all, right_sel, rpos, right_slope, right_edge, right_roi = self._detect_local_side(
            roi_gray, "right", self.last_good_rpos, self.last_good_rpos
        )

        result["local_left_roi"] = left_roi
        result["local_right_roi"] = right_roi

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
            if abs(lane_width - self.last_good_lane_width) > self.recovery_lane_width_gate_px:
                result["recovery_reason"] = "lane_width_gate_fail"
                return result
            mode = "BOTH"
            center = (lpos + rpos) // 2
            half = lane_width / 2.0
            if self.ema_half_width is None:
                self.ema_half_width = half
            else:
                alpha = self.cfg.ema_alpha
                self.ema_half_width = alpha * half + (1.0 - alpha) * self.ema_half_width
        elif lpos is not None:
            mode = "LEFT"
            lane_width = None
            if self.ema_half_width is not None:
                center = int(lpos + self.ema_half_width * self.cfg.one_lane_ratio)
            elif self.last_good_lane_width is not None:
                center = int(lpos + self.last_good_lane_width / 2.0)
            else:
                result["recovery_reason"] = "left_only_no_width"
                return result
        elif rpos is not None:
            mode = "RIGHT"
            lane_width = None
            if self.ema_half_width is not None:
                center = int(rpos - self.ema_half_width * self.cfg.one_lane_ratio)
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
            "recovery_success": 1,
            "recovery_reason": "success",
            "local_accept": 1,
            "local_reason": "success",
            "local_left_roi": left_roi,
            "local_right_roi": right_roi,
            "pred_lpos": self.last_good_lpos,
            "pred_rpos": self.last_good_rpos,
            "final_source": "LOCAL",
        })
        return result

    def _is_full_candidate_geometrically_valid(self, full_result):
        mode = full_result.get("mode")
        lpos = full_result.get("lpos")
        rpos = full_result.get("rpos")
        lane_width = full_result.get("lane_width")

        if mode == "BOTH":
            if lpos is None or rpos is None or lane_width is None:
                return False
            if int(rpos) <= int(lpos):
                return False
            lw = int(lane_width)
            if lw < self.v3_full_width_min_px or lw > self.v3_full_width_max_px:
                return False
            return True
        return False

    def _reset_tentative(self):
        self.tentative_center = None
        self.tentative_lane_width = None
        self.tentative_lpos = None
        self.tentative_rpos = None
        self.tentative_count = 0

    def _update_tentative_full_candidate(self, full_result):
        center = int(full_result["center"])
        lane_width = int(full_result["lane_width"])
        lpos = int(full_result["lpos"])
        rpos = int(full_result["rpos"])

        if self.tentative_center is None or self.tentative_lane_width is None:
            self.tentative_center = center
            self.tentative_lane_width = lane_width
            self.tentative_lpos = lpos
            self.tentative_rpos = rpos
            self.tentative_count = 1
            return self.tentative_count

        center_ok = abs(center - self.tentative_center) <= self.v3_tentative_center_gate_px
        width_ok = abs(lane_width - self.tentative_lane_width) <= self.v3_tentative_width_gate_px
        if center_ok and width_ok:
            self.tentative_center = center
            self.tentative_lane_width = lane_width
            self.tentative_lpos = lpos
            self.tentative_rpos = rpos
            self.tentative_count += 1
        else:
            self.tentative_center = center
            self.tentative_lane_width = lane_width
            self.tentative_lpos = lpos
            self.tentative_rpos = rpos
            self.tentative_count = 1
        return self.tentative_count

    def _decorate_full_recheck_result(self, full_result, baseline_mode, local_result, decision, accept):
        result = dict(full_result)
        result["baseline_mode"] = baseline_mode
        result["local_first_used"] = 1
        result["local_accept"] = int(local_result.get("local_accept", 0))
        result["full_fallback_used"] = 1
        result["local_reason"] = local_result.get("local_reason", local_result.get("recovery_reason", ""))
        result["recovery_used"] = 1
        result["recovery_success"] = int(local_result.get("recovery_success", 0))
        result["recovery_reason"] = "v3_full_recheck:%s" % decision
        result["v3_full_recheck_used"] = 1
        result["v3_mismatch"] = 1
        result["v3_tentative_count"] = self.tentative_count
        result["v3_full_recheck_accept"] = 1 if accept else 0
        result["v3_decision"] = decision
        result["local_left_roi"] = local_result.get("local_left_roi", None)
        result["local_right_roi"] = local_result.get("local_right_roi", None)
        result["pred_lpos"] = local_result.get("pred_lpos", self.last_good_lpos)
        result["pred_rpos"] = local_result.get("pred_rpos", self.last_good_rpos)
        result["final_source"] = "FULL_RECHECK_ACCEPT" if accept else "FULL_RECHECK_HOLD"
        return result

    def _process_v3_fixed(self, frame_bgr):
        full_result = self._run_full_roi_detection(frame_bgr)
        baseline_mode = full_result["mode"]

        can_try_local = (
            self.last_good_lpos is not None and
            self.last_good_rpos is not None and
            self.last_good_lane_width is not None
        )

        if not can_try_local:
            self._reset_tentative()
            full_result["baseline_mode"] = baseline_mode
            full_result["local_first_used"] = 0
            full_result["local_accept"] = 0
            full_result["full_fallback_used"] = 1
            full_result["local_reason"] = "no_last_good"
            full_result["recovery_used"] = 0
            full_result["recovery_success"] = 0
            full_result["recovery_reason"] = "no_last_good"
            full_result["v3_decision"] = "no_last_good_use_full"
            full_result["final_source"] = "FULL_NO_LAST_GOOD"
            return self._finalize_result(full_result)

        local_result = self._attempt_local_first(full_result)
        local_result["baseline_mode"] = baseline_mode
        local_result["local_first_used"] = 1
        local_result["local_accept"] = int(local_result.get("recovery_success", 0) == 1)
        local_result["local_reason"] = local_result.get("recovery_reason", "")
        local_result["full_fallback_used"] = 0
        local_result["v3_full_recheck_used"] = 0
        local_result["v3_mismatch"] = 0
        local_result["v3_tentative_count"] = self.tentative_count
        local_result["v3_full_recheck_accept"] = 0
        local_result["v3_decision"] = "local_first"

        local_ok = int(local_result.get("local_accept", 0)) == 1
        full_valid = self._is_full_candidate_geometrically_valid(full_result)

        def estimated_center_jump(candidate):
            return int(abs(int(candidate["center"]) - int(self.prev_center)))

        def estimated_lane_width_jump(candidate):
            if candidate.get("lane_width") is None or self.prev_lane_width is None:
                return None
            return int(abs(int(candidate["lane_width"]) - int(self.prev_lane_width)))

        def is_candidate_stable(candidate):
            cj = estimated_center_jump(candidate)
            lwj = estimated_lane_width_jump(candidate)
            if cj > self.center_jump_threshold:
                return False
            if lwj is not None and lwj > self.lane_width_jump_threshold:
                return False
            return True

        def is_local_both_stable(candidate):
            if candidate.get("mode") != "BOTH":
                return False
            if candidate.get("lpos") is None or candidate.get("rpos") is None or candidate.get("lane_width") is None:
                return False
            lane_width = int(candidate["lane_width"])
            if lane_width < self.v3_full_width_min_px or lane_width > self.v3_full_width_max_px:
                return False
            return is_candidate_stable(candidate)

        def accept_local(decision, full_recheck_used=0, mismatch=0, tentative_count=None):
            local_result["v3_full_recheck_used"] = int(full_recheck_used)
            local_result["v3_mismatch"] = int(mismatch)
            local_result["v3_tentative_count"] = self.tentative_count if tentative_count is None else int(tentative_count)
            local_result["v3_full_recheck_accept"] = 0
            local_result["v3_decision"] = decision
            local_result["final_source"] = "LOCAL" if local_result.get("local_accept", 0) == 1 else "LOCAL_HOLD"
            return self._finalize_result(local_result)

        def accept_full(decision, tentative_count=None):
            chosen = self._decorate_full_recheck_result(
                full_result,
                baseline_mode,
                local_result,
                decision=decision,
                accept=True,
            )
            chosen["v3_tentative_count"] = self.tentative_count if tentative_count is None else int(tentative_count)
            return self._finalize_result(chosen)

        # local BOTH가 안정적이면 v2처럼 그대로 유지한다.
        if local_ok and is_local_both_stable(local_result):
            self._reset_tentative()
            return accept_local("accept_local_stable_both")

        # local 성공이지만 단일 차선/불안정이면 full ROI BOTH 후보만 제한적으로 개입시킨다.
        if local_ok:
            local_mode = str(local_result.get("mode", ""))
            if full_valid:
                center_diff = abs(int(local_result["center"]) - int(full_result["center"]))
                width_diff = None
                if local_result.get("lane_width") is not None and full_result.get("lane_width") is not None:
                    width_diff = abs(int(local_result["lane_width"]) - int(full_result["lane_width"]))

                mismatch = center_diff > self.v3_mismatch_center_gate_px
                if width_diff is not None and width_diff > self.v3_mismatch_lane_width_gate_px:
                    mismatch = True

                if local_mode != "BOTH":
                    if is_candidate_stable(full_result):
                        self._reset_tentative()
                        return accept_full("accept_full_both_because_local_not_both")
                    tentative_count = self._update_tentative_full_candidate(full_result)
                    if tentative_count >= self.v3_tentative_accept_frames:
                        return accept_full("accept_full_after_tentative_local_not_both", tentative_count)
                    return accept_local("hold_local_not_both_wait_full_consistency", 1, 1, tentative_count)

                if mismatch:
                    tentative_count = self._update_tentative_full_candidate(full_result)
                    if tentative_count >= self.v3_tentative_accept_frames and is_candidate_stable(full_result):
                        return accept_full("accept_full_after_tentative_local_unstable", tentative_count)
                    return accept_local("hold_local_both_unstable_wait_full_consistency", 1, 1, tentative_count)

                self._reset_tentative()
                return accept_local("accept_local_full_similar", 1, 0)

            self._reset_tentative()
            return accept_local("accept_local_no_valid_full")

        # local 실패 시 full ROI fallback
        self._reset_tentative()
        full_result["baseline_mode"] = baseline_mode
        full_result["local_first_used"] = 1
        full_result["local_accept"] = 0
        full_result["full_fallback_used"] = 1
        full_result["local_reason"] = local_result.get("recovery_reason", "local_fail")
        full_result["recovery_used"] = 1
        full_result["recovery_success"] = 0
        full_result["recovery_reason"] = "fallback_after_local_fail:%s" % local_result.get("recovery_reason", "local_fail")
        full_result["v3_full_recheck_used"] = 1
        full_result["v3_mismatch"] = 0
        full_result["v3_tentative_count"] = 0
        full_result["v3_full_recheck_accept"] = 0
        full_result["local_left_roi"] = local_result.get("local_left_roi", None)
        full_result["local_right_roi"] = local_result.get("local_right_roi", None)
        full_result["pred_lpos"] = local_result.get("pred_lpos", self.last_good_lpos)
        full_result["pred_rpos"] = local_result.get("pred_rpos", self.last_good_rpos)
        full_result["final_source"] = "FULL_FALLBACK"
        full_result["v3_decision"] = "accept_full_both_after_local_fail" if full_valid else "fallback_full_after_local_fail"
        return self._finalize_result(full_result)

    def process(self, frame):
        """프레임 -> (center, mode, 좌/우 선분 리스트, lpos, rpos, corner, left_slope)

        v3_fixed live 주행용:
        - full ROI baseline은 매 프레임 계산
        - last_good 기반 local ROI를 우선 사용
        - local 결과가 불안정할 때만 full ROI 재검증 개입
        """
        if frame is None or frame.size == 0:
            return self.prev_center, "NONE", [], [], None, None, "STRAIGHT", None

        result = self._process_v3_fixed(frame)
        return (
            int(result.get("center", self.prev_center)),
            result.get("mode", "NONE"),
            result.get("selected_left", []),
            result.get("selected_right", []),
            result.get("lpos", None),
            result.get("rpos", None),
            result.get("corner", "STRAIGHT"),
            result.get("left_slope", None),
        )

    def draw_debug(self, frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos, corner, left_slope):
        y = self.cfg.offset + self.cfg.gap // 2
        y0 = self.cfg.offset
        y1 = self.cfg.offset + self.cfg.gap

        # full ROI
        cv2.rectangle(frame, (0, y0), (self.cfg.width - 1, y1), (0, 255, 0), 2)

        result = self.last_debug_result if self.last_debug_result is not None else {}

        # local ROI 박스: left=초록, right=보라
        local_left_roi = result.get("local_left_roi", None)
        local_right_roi = result.get("local_right_roi", None)
        if local_left_roi is not None:
            lx0, lx1 = local_left_roi
            cv2.rectangle(frame, (int(lx0), y0), (int(lx1), y1), (0, 180, 0), 2)
        if local_right_roi is not None:
            rx0, rx1 = local_right_roi
            cv2.rectangle(frame, (int(rx0), y0), (int(rx1), y1), (180, 0, 180), 2)

        # prediction 위치 표시
        pred_lpos = result.get("pred_lpos", None)
        pred_rpos = result.get("pred_rpos", None)
        if pred_lpos is not None:
            cv2.line(frame, (int(pred_lpos), y0), (int(pred_lpos), y1), (0, 120, 0), 1)
            cv2.putText(frame, "predL", (int(pred_lpos) + 3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)
        if pred_rpos is not None:
            cv2.line(frame, (int(pred_rpos), y0), (int(pred_rpos), y1), (120, 0, 120), 1)
            cv2.putText(frame, "predR", (int(pred_rpos) + 3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 0, 120), 1)

        # 선택된 차선 선분
        for x1, y1l, x2, y2l in left:
            cv2.line(frame, (x1, y1l + self.cfg.offset), (x2, y2l + self.cfg.offset), (0, 0, 255), 2)
        for x1, y1r, x2, y2r in right:
            cv2.line(frame, (x1, y1r + self.cfg.offset), (x2, y2r + self.cfg.offset), (255, 0, 0), 2)

        # 차선 위치 점
        if lpos is not None:
            cv2.circle(frame, (int(lpos), y), 6, (0, 0, 0), -1)
        if rpos is not None:
            cv2.circle(frame, (int(rpos), y), 6, (0, 0, 0), -1)

        cv2.circle(frame, (int(cam_center), y), 6, (255, 128, 0), -1)
        if lidar_c is not None:
            cv2.circle(frame, (int(lidar_c), y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (int(center), y), 8, (0, 255, 255), 2)
        cv2.circle(frame, (self.cfg.width // 2, y), 6, (255, 255, 255), -1)

        source = result.get("final_source", "-")
        decision = result.get("v3_decision", "-")
        base_mode = result.get("baseline_mode", "-")
        text1 = "mode=%s  base=%s  src=%s" % (mode, base_mode, source)
        text2 = "decision=%s" % decision
        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, text2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        slope_str = "Lslope: %.2f" % left_slope if left_slope is not None else "Lslope: -"
        cv2.putText(frame, slope_str, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        label = {"STRAIGHT": "STRAIGHT", "LEFT": "<< LEFT"}.get(corner, corner)
        color = (0, 255, 0) if corner == "STRAIGHT" else (0, 165, 255)
        cv2.putText(frame, label, (self.cfg.width // 2 - 60, self.cfg.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('lane_drive', frame)
        cv2.waitKey(1)

    def shutdown(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
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
    

class ARtagDetector:
    def __init__(self):
        self.last_seen = 0.0
        self.hold_sec = 0.5

        self.forward_dist = None   # 전방 거리, 보통 z값
        self.detect_dist = 0.5     # 실제 3차원 거리

        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.callback, queue_size=1)

    def callback(self, msg):
        if len(msg.markers) > 0:
            self.last_seen = rospy.get_time()
            # AR 마커의 위치 정보 추출
            marker = msg.markers[0]
            # self.distance = marker.pose.position.z
            self.forward_dist = marker.pose.position.x

    @property
    def ar_detected(self):
        return (rospy.get_time() - self.last_seen) < self.hold_sec
    
    def get_distance(self):
        if self.forward_dist <= self.detect_dist:
            return True
        return False