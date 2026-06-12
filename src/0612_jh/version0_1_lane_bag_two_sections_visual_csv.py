#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version0_1_lane_bag_two_sections_visual_csv.py

목적:
    하나의 rosbag에서 사용자가 코드 안에 지정한 여러 시간 구간만 골라서,
    버전 0.1 차선 검출 알고리즘을 검증한다.
    버전 0.1은 full ROI 구조는 유지하고, CLAHE + adaptive Canny만 적용한다.

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

    python3 version0_1_lane_bag_two_sections_visual_csv.py \
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
#         "name": "full_bag_check",
#         "start_sec": 0.0,
#         "end_sec": 9999.0,
#     },
# ]
TIME_SEGMENTS = [
    {
        "name": "lap1_problem_section_v0_1",
        "start_sec": 14.0,
        "end_sec": 80.0,
    },
    {
        "name": "lap2_problem_section_v0_1",
        "start_sec": 104.0,
        "end_sec": 115.0,
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

    # Version 0.1 preprocessing parameters
    clahe_clip: float = 2.0
    adaptive_canny_sigma: float = 0.33
    adaptive_canny_low_min: int = 15
    adaptive_canny_high_min: int = 45
    adaptive_canny_high_max: int = 180


class BaselineLaneDetector:
    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.prev_center = cfg.width // 2
        self.ema_half_width: Optional[float] = None
        self.prev_lane_width: Optional[float] = None
        self.fail_count = 0

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

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        cfg = self.cfg

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        y0 = max(0, int(cfg.offset))
        y1 = min(gray.shape[0], int(cfg.offset + cfg.gap))
        x0 = 0
        x1 = min(gray.shape[1], int(cfg.width))

        roi_gray = gray[y0:y1, x0:x1]

        # Version 0.1: full ROI는 그대로 유지하고, ROI 내부에만 CLAHE + adaptive Canny 적용
        if roi_gray.size > 0:
            clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8))
            roi_enhanced = clahe.apply(roi_gray)
            blur = cv2.GaussianBlur(roi_enhanced, (5, 5), 0)
            med = float(np.median(blur))
            low = int(max(cfg.adaptive_canny_low_min, (1.0 - cfg.adaptive_canny_sigma) * med))
            high = int(min(cfg.adaptive_canny_high_max, max(cfg.adaptive_canny_high_min, (1.0 + cfg.adaptive_canny_sigma) * med)))
            if high <= low:
                high = low + 30
            roi_edge = cv2.Canny(blur, low, high)
        else:
            roi_edge = np.zeros_like(roi_gray)

        edge = np.zeros_like(gray)
        edge[y0:y1, x0:x1] = roi_edge

        lines = cv2.HoughLinesP(
            roi_edge,
            1,
            math.pi / 180.0,
            cfg.hough_threshold,
            minLineLength=cfg.hough_min_len,
            maxLineGap=cfg.hough_max_gap,
        )

        hough_line_count = 0 if lines is None else int(len(lines))
        all_lines: List[Tuple[int, int, int, int]] = []
        selected_left: List[Tuple[int, int, int, int]] = []
        selected_right: List[Tuple[int, int, int, int]] = []

        lpos: Optional[int] = None
        rpos: Optional[int] = None
        lane_width: Optional[int] = None
        left_slope: Optional[float] = None
        right_slope: Optional[float] = None
        corner = "STRAIGHT"

        if lines is None:
            mode = "NONE"
            center = self.prev_center
            left_candidates: List[Tuple[int, int, int, int]] = []
            right_candidates: List[Tuple[int, int, int, int]] = []
        else:
            all_lines = [tuple(map(int, line[0])) for line in lines]
            left_candidates, right_candidates = self.divide_left_right(lines)

            if len(left_candidates) > 0:
                selected_left = [max(left_candidates, key=self._mid_x)]
                lpos = self.get_pos(selected_left)
                left_slope = self._slope(selected_left[0])

            if len(right_candidates) > 0:
                selected_right = [min(right_candidates, key=self._mid_x)]
                rpos = self.get_pos(selected_right)
                right_slope = self._slope(selected_right[0])

            if left_slope is not None:
                dev = left_slope - cfg.corner_left_base
                if dev < -cfg.corner_slope_thresh:
                    corner = "LEFT"

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

            if corner == "LEFT":
                center -= cfg.corner_shift_px

        center = int(max(0, min(cfg.width - 1, center)))
        center_error = int(center - cfg.width // 2)

        if mode == "NONE":
            self.fail_count += 1
        else:
            self.fail_count = 0

        center_jump = int(abs(center - self.prev_center))
        lane_width_jump: Optional[int] = None
        if lane_width is not None and self.prev_lane_width is not None:
            lane_width_jump = int(abs(lane_width - self.prev_lane_width))

        suspicious_reasons: List[str] = []
        if mode == "NONE":
            suspicious_reasons.append("mode_NONE")
        if hough_line_count == 0:
            suspicious_reasons.append("hough_0")
        if center_jump > cfg.center_jump_threshold:
            suspicious_reasons.append("center_jump")
        if lane_width_jump is not None and lane_width_jump > cfg.lane_width_jump_threshold:
            suspicious_reasons.append("lane_width_jump")

        self.prev_center = center
        if lane_width is not None:
            self.prev_lane_width = float(lane_width)

        roi_mean = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0
        roi_std = float(np.std(roi_gray)) if roi_gray.size > 0 else 0.0
        roi_min = int(np.min(roi_gray)) if roi_gray.size > 0 else 0
        roi_max = int(np.max(roi_gray)) if roi_gray.size > 0 else 0
        edge_pixel_count = int(np.count_nonzero(roi_edge)) if roi_edge.size > 0 else 0

        return {
            "gray": gray,
            "edge": edge,
            "roi_gray": roi_gray,
            "roi_edge": roi_edge,
            "all_lines": all_lines,
            "selected_left": selected_left,
            "selected_right": selected_right,
            "left_candidates": left_candidates if lines is not None else [],
            "right_candidates": right_candidates if lines is not None else [],
            "lpos": lpos,
            "rpos": rpos,
            "lane_width": lane_width,
            "center": center,
            "center_error": center_error,
            "mode": mode,
            "corner": corner,
            "left_slope": left_slope,
            "right_slope": right_slope,
            "hough_line_count": hough_line_count,
            "left_line_count": len(left_candidates) if lines is not None else 0,
            "right_line_count": len(right_candidates) if lines is not None else 0,
            "roi_mean": roi_mean,
            "roi_std": roi_std,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "edge_pixel_count": edge_pixel_count,
            "fail_count": self.fail_count,
            "center_jump": center_jump,
            "lane_width_jump": lane_width_jump,
            "suspicious_reasons": ";".join(suspicious_reasons),
        }


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

    if result["suspicious_reasons"]:
        draw_text(overlay, "WARN: {}".format(result["suspicious_reasons"]), (10, 125))

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
        "corner",
        "fail_count",
        "center_jump",
        "lane_width_jump",
        "suspicious_reasons",
    ]


def open_segment_csv(output_dir: str, segment_name: str) -> Tuple[Any, csv.DictWriter, str]:
    filename = safe_filename(segment_name) + ".csv"
    csv_path = os.path.join(output_dir, filename)
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
    writer.writeheader()
    return f, writer, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rosbag 한 개에서 코드 내부 TIME_SEGMENTS 구간만 버전 0.1 CLAHE + adaptive Canny 차선 검출 분석 + 구간별 CSV 저장"
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

    # Version 0.1 preprocessing parameters
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--adaptive-canny-sigma", type=float, default=0.33)
    parser.add_argument("--adaptive-canny-low-min", type=int, default=15)
    parser.add_argument("--adaptive-canny-high-min", type=int, default=45)
    parser.add_argument("--adaptive-canny-high-max", type=int, default=180)

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
        cv2.imshow("baseline section overlay - no save", resize_for_display(overlay, display_scale))

    if view in ["debug", "both"]:
        debug_grid = make_debug_grid(frame_bgr, overlay, result)
        cv2.imshow("baseline section debug - no save", resize_for_display(debug_grid, display_scale))

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
        clahe_clip=args.clahe_clip,
        adaptive_canny_sigma=args.adaptive_canny_sigma,
        adaptive_canny_low_min=args.adaptive_canny_low_min,
        adaptive_canny_high_min=args.adaptive_canny_high_min,
        adaptive_canny_high_max=args.adaptive_canny_high_max,
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
                        "corner": result["corner"],
                        "fail_count": result["fail_count"],
                        "center_jump": result["center_jump"],
                        "lane_width_jump": "" if result["lane_width_jump"] is None else result["lane_width_jump"],
                        "suspicious_reasons": result["suspicious_reasons"],
                    }
                    seg.writer.writerow(row)

                    if seg.processed_count % 50 == 0:
                        seg.csv_file.flush()
                        print("[INFO] segment={} | processed={} | frame_idx={} | rel_t={:.3f}s | mode={} | center={} | roi_mean={:.1f} | hough={}".format(
                            seg.name,
                            seg.processed_count,
                            total_msg_count,
                            rel_time,
                            result["mode"],
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
    print("\nmp4/png 저장       : 없음")


if __name__ == "__main__":
    main()
