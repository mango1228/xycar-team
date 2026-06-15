#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_lane_bag_analyzer.py

목적:
    rosbag에 저장된 /usb_cam/image_raw 카메라 raw 이미지에 대해
    현재 my_motor/src/lane_drive.py의 기본 차선 검출 알고리즘을 offline으로 재현하고,
    프레임별 검출 결과를 영상/CSV/이미지로 저장한다.

검증 대상:
    - 버전 1/2/3 개선안이 아니라 현재 기본 알고리즘
    - BGR -> Gray -> GaussianBlur -> Canny -> 하단 ROI -> HoughLinesP
    - slope 기준 left/right 분류
    - lpos/rpos/center/mode 계산

출력:
    output_dir/
      lane_detection_log.csv
      summary.json
      overlay_baseline.mp4
      debug_grid_baseline.mp4
      failure_frames/
        frame_000123_overlay.png
        frame_000123_debug.png
        ...

실행 예시:
    python3 baseline_lane_bag_analyzer.py \
        --bag /path/to/raw_data.bag \
        --output-dir /path/to/lane_bag_analysis \
        --topic /usb_cam/image_raw

필요 환경:
    ROS Noetic 환경에서 실행 권장
    source /opt/ros/noetic/setup.bash
    source ~/xycar_ws/devel/setup.bash   # cv_bridge가 정상 import되지 않으면 필요

필요 패키지:
    rosbag, sensor_msgs, cv_bridge, opencv-python, numpy
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

try:
    import rosbag
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CompressedImage
except Exception as exc:
    print("[ERROR] ROS 관련 모듈 import 실패")
    print("        ROS 환경을 source 했는지 확인하세요.")
    print("        예: source /opt/ros/noetic/setup.bash")
    print("        예: source ~/xycar_ws/devel/setup.bash")
    print("        원인: {}".format(exc))
    sys.exit(1)


@dataclass
class BaselineConfig:
    # 기존 config.py / xycar_drive.launch 기준 기본값
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

    # 분석용 추가 파라미터
    center_jump_threshold: int = 80
    lane_width_jump_threshold: int = 100
    save_failure_context: bool = True


class BaselineLaneDetector:
    """lane_drive.py의 ImageProcessor.process()를 offline 분석용으로 재현한 클래스."""

    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.ema_half_width: Optional[float] = None
        self.prev_center: int = self.cfg.width // 2
        self.prev_lane_width: Optional[float] = None
        self.fail_count: int = 0

    def divide_left_right(self, lines: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        left: List[Tuple[int, int, int, int]] = []
        right: List[Tuple[int, int, int, int]] = []

        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            if x2 == x1:
                continue

            slope = float(y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.1 or abs(slope) > 10:
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

        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)

        n = len(lines)
        if n == 0:
            return None

        x_avg = x_sum / (n * 2)
        y_avg = y_sum / (n * 2)
        m = m_sum / n

        if abs(m) < 1e-12:
            return None

        b = y_avg - m * x_avg
        return int((self.cfg.gap / 2 - b) / m)

    @staticmethod
    def _line_slope(line: Tuple[int, int, int, int]) -> Optional[float]:
        x1, y1, x2, y2 = line
        if x2 == x1:
            return None
        return float(y2 - y1) / float(x2 - x1)

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        기존 lane_drive.py의 기본 차선 검출 흐름을 재현한다.
        반환 dict에는 CSV/시각화에 필요한 중간 결과를 포함한다.
        """
        cfg = self.cfg

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)

        y0 = max(0, int(cfg.offset))
        y1 = min(edge.shape[0], int(cfg.offset + cfg.gap))
        x0 = 0
        x1 = min(edge.shape[1], int(cfg.width))
        roi_edge = edge[y0:y1, x0:x1]
        roi_gray = gray[y0:y1, x0:x1]
        roi_blur = blur[y0:y1, x0:x1]

        lines = cv2.HoughLinesP(
            roi_edge,
            1,
            math.pi / 180,
            cfg.hough_threshold,
            minLineLength=cfg.hough_min_len,
            maxLineGap=cfg.hough_max_gap,
        )

        raw_hough_count = 0 if lines is None else int(len(lines))
        all_lines: List[Tuple[int, int, int, int]] = []
        if lines is not None:
            all_lines = [tuple(map(int, line[0])) for line in lines]

        if lines is None:
            mode = "NONE"
            center = self.prev_center
            left: List[Tuple[int, int, int, int]] = []
            right: List[Tuple[int, int, int, int]] = []
            left_candidates: List[Tuple[int, int, int, int]] = []
            right_candidates: List[Tuple[int, int, int, int]] = []
            lpos = None
            rpos = None
            lane_width = None
            corner = "STRAIGHT"
            left_slope = None
            right_slope = None
            self.fail_count += 1
        else:
            left_candidates, right_candidates = self.divide_left_right(lines)

            # lane_drive.py와 동일: 중앙에 가장 가까운 선분 1개만 선택
            mid = lambda line: (line[0] + line[2]) / 2.0
            left = [max(left_candidates, key=mid)] if left_candidates else []
            right = [min(right_candidates, key=mid)] if right_candidates else []

            lpos = self.get_pos(left)
            rpos = self.get_pos(right)

            corner = "STRAIGHT"
            left_slope = self._line_slope(left[0]) if left else None
            right_slope = self._line_slope(right[0]) if right else None

            if left_slope is not None:
                dev = left_slope - cfg.corner_left_base
                if dev < -cfg.corner_slope_thresh:
                    corner = "LEFT"

            if lpos is not None and rpos is not None:
                center = (lpos + rpos) // 2
                lane_width = float(rpos - lpos)
                half = lane_width / 2.0
                if self.ema_half_width is None:
                    self.ema_half_width = half
                else:
                    self.ema_half_width = cfg.ema_alpha * half + (1.0 - cfg.ema_alpha) * self.ema_half_width
                mode = "BOTH"
            elif lpos is not None:
                lane_width = None
                if self.ema_half_width is not None:
                    center = int(lpos + self.ema_half_width * cfg.one_lane_ratio)
                else:
                    center = self.prev_center
                mode = "LEFT"
            elif rpos is not None:
                lane_width = None
                if self.ema_half_width is not None:
                    center = int(rpos - self.ema_half_width * cfg.one_lane_ratio)
                else:
                    center = self.prev_center
                mode = "RIGHT"
            else:
                lane_width = None
                center = self.prev_center
                mode = "NONE"

            if corner == "LEFT":
                center -= cfg.corner_shift_px

            center = int(max(0, min(cfg.width - 1, center)))

            if mode == "NONE":
                self.fail_count += 1
            else:
                self.fail_count = 0

        cam_center = cfg.width // 2
        center_error = int(center - cam_center)
        center_jump = int(center - self.prev_center)

        lane_width_jump = None
        if lane_width is not None and self.prev_lane_width is not None:
            lane_width_jump = float(lane_width - self.prev_lane_width)

        if lane_width is not None:
            self.prev_lane_width = lane_width

        self.prev_center = center

        edge_pixel_count = int(np.count_nonzero(roi_edge))
        roi_mean = float(np.mean(roi_gray)) if roi_gray.size else 0.0
        roi_std = float(np.std(roi_gray)) if roi_gray.size else 0.0
        roi_min = int(np.min(roi_gray)) if roi_gray.size else 0
        roi_max = int(np.max(roi_gray)) if roi_gray.size else 0

        suspicious = False
        suspicious_reasons: List[str] = []
        if mode == "NONE":
            suspicious = True
            suspicious_reasons.append("mode_NONE")
        if raw_hough_count == 0:
            suspicious = True
            suspicious_reasons.append("hough_0")
        if abs(center_jump) > cfg.center_jump_threshold:
            suspicious = True
            suspicious_reasons.append("center_jump")
        if lane_width_jump is not None and abs(lane_width_jump) > cfg.lane_width_jump_threshold:
            suspicious = True
            suspicious_reasons.append("lane_width_jump")

        return {
            "gray": gray,
            "blur": blur,
            "edge": edge,
            "roi_gray": roi_gray,
            "roi_blur": roi_blur,
            "roi_edge": roi_edge,
            "all_lines": all_lines,
            "left_candidates": left_candidates,
            "right_candidates": right_candidates,
            "left": left,
            "right": right,
            "lpos": lpos,
            "rpos": rpos,
            "lane_width": lane_width,
            "center": center,
            "cam_center": cam_center,
            "center_error": center_error,
            "center_jump": center_jump,
            "lane_width_jump": lane_width_jump,
            "mode": mode,
            "corner": corner,
            "left_slope": left_slope,
            "right_slope": right_slope,
            "fail_count": self.fail_count,
            "roi_mean": roi_mean,
            "roi_std": roi_std,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "edge_pixel_count": edge_pixel_count,
            "hough_line_count": raw_hough_count,
            "left_line_count": len(left_candidates),
            "right_line_count": len(right_candidates),
            "suspicious": suspicious,
            "suspicious_reasons": ";".join(suspicious_reasons),
        }


class ImageConverter:
    def __init__(self):
        self.bridge = CvBridge()

    def to_bgr8(self, msg: Any) -> np.ndarray:
        """sensor_msgs/Image 또는 CompressedImage를 OpenCV BGR 이미지로 변환."""
        if isinstance(msg, Image):
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("CompressedImage decode 실패")
            return img

        # rosbag에서 타입 비교가 꼬이는 경우를 대비한 fallback
        if hasattr(msg, "format") and hasattr(msg, "data") and msg.__class__.__name__ == "CompressedImage":
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("CompressedImage decode 실패")
            return img

        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def safe_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))


def draw_overlay(frame_bgr: np.ndarray, result: Dict[str, Any], cfg: BaselineConfig, frame_idx: int, timestamp: float) -> np.ndarray:
    img = frame_bgr.copy()

    h, w = img.shape[:2]
    y0 = max(0, int(cfg.offset))
    y1 = min(h - 1, int(cfg.offset + cfg.gap))
    roi_mid_y = int(cfg.offset + cfg.gap // 2)

    # ROI 박스
    cv2.rectangle(img, (0, y0), (min(cfg.width - 1, w - 1), y1), (0, 255, 0), 2)

    # 전체 Hough 후보: 회색
    for x1, y1r, x2, y2r in result["all_lines"]:
        cv2.line(img, (x1, y1r + cfg.offset), (x2, y2r + cfg.offset), (120, 120, 120), 1)

    # 좌/우 후보: 약한 색
    for x1, y1r, x2, y2r in result["left_candidates"]:
        cv2.line(img, (x1, y1r + cfg.offset), (x2, y2r + cfg.offset), (0, 80, 255), 1)
    for x1, y1r, x2, y2r in result["right_candidates"]:
        cv2.line(img, (x1, y1r + cfg.offset), (x2, y2r + cfg.offset), (255, 80, 0), 1)

    # 최종 선택 선분
    for x1, y1r, x2, y2r in result["left"]:
        cv2.line(img, (x1, y1r + cfg.offset), (x2, y2r + cfg.offset), (0, 0, 255), 3)
    for x1, y1r, x2, y2r in result["right"]:
        cv2.line(img, (x1, y1r + cfg.offset), (x2, y2r + cfg.offset), (255, 0, 0), 3)

    # lpos/rpos/center/camera center
    lpos = result["lpos"]
    rpos = result["rpos"]
    center = int(result["center"])
    cam_center = int(result["cam_center"])

    if lpos is not None:
        cv2.circle(img, (int(lpos), roi_mid_y), 7, (0, 0, 0), -1)
    if rpos is not None:
        cv2.circle(img, (int(rpos), roi_mid_y), 7, (0, 0, 0), -1)

    cv2.line(img, (cam_center, y0), (cam_center, y0 + cfg.gap), (255, 255, 255), 2)
    cv2.circle(img, (center, roi_mid_y), 9, (0, 255, 255), 2)

    # 텍스트 정보
    lines = [
        "frame={}  t={:.3f}".format(frame_idx, timestamp),
        "mode={}  corner={}  fail={}".format(result["mode"], result["corner"], result["fail_count"]),
        "roi_mean={:.1f} std={:.1f} edge={} hough={}".format(
            result["roi_mean"], result["roi_std"], result["edge_pixel_count"], result["hough_line_count"]
        ),
        "lpos={} rpos={} center={} err={}".format(
            result["lpos"], result["rpos"], result["center"], result["center_error"]
        ),
    ]

    x_text = 10
    y_text = 25
    for i, text in enumerate(lines):
        y = y_text + i * 27
        cv2.putText(img, text, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    if result["suspicious"]:
        warning = "SUSPICIOUS: {}".format(result["suspicious_reasons"])
        cv2.putText(img, warning, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img, warning, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def to_bgr_panel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def label_panel(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_hough_roi_panel(result: Dict[str, Any], cfg: BaselineConfig) -> np.ndarray:
    roi = to_bgr_panel(result["roi_edge"])

    for x1, y1, x2, y2 in result["all_lines"]:
        cv2.line(roi, (x1, y1), (x2, y2), (120, 120, 120), 1)
    for x1, y1, x2, y2 in result["left"]:
        cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in result["right"]:
        cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

    mid_y = cfg.gap // 2
    if result["lpos"] is not None:
        cv2.circle(roi, (int(result["lpos"]), mid_y), 5, (0, 0, 255), -1)
    if result["rpos"] is not None:
        cv2.circle(roi, (int(result["rpos"]), mid_y), 5, (255, 0, 0), -1)
    cv2.circle(roi, (int(result["center"]), mid_y), 6, (0, 255, 255), 2)

    return roi


def make_debug_grid(frame_bgr: np.ndarray, overlay: np.ndarray, result: Dict[str, Any], cfg: BaselineConfig) -> np.ndarray:
    panel_w = 320
    panel_h = 240

    original = cv2.resize(frame_bgr, (panel_w, panel_h))
    gray = cv2.resize(to_bgr_panel(result["gray"]), (panel_w, panel_h))
    edge = cv2.resize(to_bgr_panel(result["edge"]), (panel_w, panel_h))
    roi_gray = cv2.resize(to_bgr_panel(result["roi_gray"]), (panel_w, panel_h))
    roi_hough = cv2.resize(draw_hough_roi_panel(result, cfg), (panel_w, panel_h))
    overlay_small = cv2.resize(overlay, (panel_w, panel_h))

    panels = [
        label_panel(original, "Original"),
        label_panel(gray, "Gray"),
        label_panel(edge, "Canny edge"),
        label_panel(roi_gray, "ROI gray"),
        label_panel(roi_hough, "ROI Hough"),
        label_panel(overlay_small, "Final overlay"),
    ]

    top = np.hstack(panels[0:3])
    bottom = np.hstack(panels[3:6])
    grid = np.vstack([top, bottom])

    info = "mode={} | roi_mean={:.1f} | edge={} | hough={} | lpos={} | rpos={} | center={} | fail={}".format(
        result["mode"],
        result["roi_mean"],
        result["edge_pixel_count"],
        result["hough_line_count"],
        result["lpos"],
        result["rpos"],
        result["center"],
        result["fail_count"],
    )
    cv2.rectangle(grid, (0, grid.shape[0] - 30), (grid.shape[1], grid.shape[0]), (0, 0, 0), -1)
    cv2.putText(grid, info, (8, grid.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return grid


def write_csv_header(csv_writer: csv.writer) -> None:
    csv_writer.writerow([
        "frame_idx",
        "timestamp_sec",
        "topic",
        "image_width",
        "image_height",
        "roi_mean",
        "roi_std",
        "roi_min",
        "roi_max",
        "edge_pixel_count",
        "hough_line_count",
        "left_line_count",
        "right_line_count",
        "selected_left_slope",
        "selected_right_slope",
        "lpos",
        "rpos",
        "lane_width",
        "center",
        "image_center",
        "center_error",
        "center_jump",
        "lane_width_jump",
        "mode",
        "corner",
        "fail_count",
        "suspicious",
        "suspicious_reasons",
    ])


def write_csv_row(csv_writer: csv.writer, frame_idx: int, timestamp_sec: float, topic: str, frame: np.ndarray, result: Dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    csv_writer.writerow([
        frame_idx,
        "{:.6f}".format(timestamp_sec),
        topic,
        w,
        h,
        "{:.6f}".format(result["roi_mean"]),
        "{:.6f}".format(result["roi_std"]),
        result["roi_min"],
        result["roi_max"],
        result["edge_pixel_count"],
        result["hough_line_count"],
        result["left_line_count"],
        result["right_line_count"],
        safe_float(result["left_slope"]),
        safe_float(result["right_slope"]),
        safe_int(result["lpos"]),
        safe_int(result["rpos"]),
        safe_float(result["lane_width"]),
        result["center"],
        result["cam_center"],
        result["center_error"],
        result["center_jump"],
        safe_float(result["lane_width_jump"]),
        result["mode"],
        result["corner"],
        result["fail_count"],
        int(result["suspicious"]),
        result["suspicious_reasons"],
    ])


def open_video_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("VideoWriter open 실패: {}".format(path))
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="카메라 raw bag 기반 기본 차선 검출 offline analyzer")

    parser.add_argument("--bag", required=True, help="입력 rosbag 파일 경로")
    parser.add_argument("--topic", default="/usb_cam/image_raw", help="카메라 이미지 토픽 이름")
    parser.add_argument("--output-dir", default="lane_bag_analysis", help="결과 저장 폴더")

    parser.add_argument("--fps", type=float, default=20.0, help="출력 영상 FPS")
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수. 0이면 전체")
    parser.add_argument("--every-n", type=int, default=1, help="N프레임마다 1개 처리. 기본 1")

    # baseline 파라미터 override
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
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--one-lane-ratio", type=float, default=1.0)
    parser.add_argument("--corner-left-base", type=float, default=-0.75)
    parser.add_argument("--corner-slope-thresh", type=float, default=0.4)
    parser.add_argument("--corner-shift-px", type=int, default=70)

    parser.add_argument("--center-jump-threshold", type=int, default=80, help="문제 프레임 저장 기준 center jump [px]")
    parser.add_argument("--lane-width-jump-threshold", type=int, default=100, help="문제 프레임 저장 기준 lane width jump [px]")
    parser.add_argument("--no-video", action="store_true", help="영상 저장 비활성화")
    parser.add_argument("--no-failure-images", action="store_true", help="문제 프레임 이미지 저장 비활성화")

    return parser.parse_args()


def make_config(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
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
        ema_alpha=args.ema_alpha,
        one_lane_ratio=args.one_lane_ratio,
        corner_left_base=args.corner_left_base,
        corner_slope_thresh=args.corner_slope_thresh,
        corner_shift_px=args.corner_shift_px,
        center_jump_threshold=args.center_jump_threshold,
        lane_width_jump_threshold=args.lane_width_jump_threshold,
        save_failure_context=not args.no_failure_images,
    )


def analyze_bag(args: argparse.Namespace) -> None:
    cfg = make_config(args)

    if not os.path.isfile(args.bag):
        raise FileNotFoundError("bag 파일을 찾을 수 없습니다: {}".format(args.bag))

    ensure_dir(args.output_dir)
    failure_dir = os.path.join(args.output_dir, "failure_frames")
    if not args.no_failure_images:
        ensure_dir(failure_dir)

    csv_path = os.path.join(args.output_dir, "lane_detection_log.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    overlay_video_path = os.path.join(args.output_dir, "overlay_baseline.mp4")
    debug_video_path = os.path.join(args.output_dir, "debug_grid_baseline.mp4")

    detector = BaselineLaneDetector(cfg)
    converter = ImageConverter()

    overlay_writer: Optional[cv2.VideoWriter] = None
    debug_writer: Optional[cv2.VideoWriter] = None

    total_frames = 0
    processed_frames = 0
    suspicious_frames = 0
    mode_counts = {"BOTH": 0, "LEFT": 0, "RIGHT": 0, "NONE": 0}
    max_fail_count = 0
    max_abs_center_jump = 0
    max_abs_lane_width_jump = 0.0

    print("[INFO] bag 분석 시작")
    print("[INFO] bag        : {}".format(args.bag))
    print("[INFO] topic      : {}".format(args.topic))
    print("[INFO] output_dir : {}".format(args.output_dir))
    print("[INFO] baseline cfg: {}".format(asdict(cfg)))

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

        with rosbag.Bag(args.bag, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[args.topic]):
                total_frames += 1

                if args.every_n <= 0:
                    args.every_n = 1
                if (total_frames - 1) % args.every_n != 0:
                    continue

                if args.max_frames > 0 and processed_frames >= args.max_frames:
                    break

                try:
                    frame = converter.to_bgr8(msg)
                except Exception as exc:
                    print("[WARN] frame 변환 실패: frame={} error={}".format(total_frames - 1, exc))
                    continue

                if frame is None or frame.size == 0:
                    print("[WARN] 빈 이미지 frame skip: frame={}".format(total_frames - 1))
                    continue

                timestamp_sec = t.to_sec()
                result = detector.process(frame)
                overlay = draw_overlay(frame, result, cfg, processed_frames, timestamp_sec)
                debug_grid = make_debug_grid(frame, overlay, result, cfg)

                if overlay_writer is None and not args.no_video:
                    oh, ow = overlay.shape[:2]
                    dh, dw = debug_grid.shape[:2]
                    overlay_writer = open_video_writer(overlay_video_path, args.fps, (ow, oh))
                    debug_writer = open_video_writer(debug_video_path, args.fps, (dw, dh))

                if overlay_writer is not None:
                    overlay_writer.write(overlay)
                if debug_writer is not None:
                    debug_writer.write(debug_grid)

                write_csv_row(csv_writer, processed_frames, timestamp_sec, topic, frame, result)

                mode = result["mode"]
                if mode not in mode_counts:
                    mode_counts[mode] = 0
                mode_counts[mode] += 1

                if result["suspicious"]:
                    suspicious_frames += 1
                    if not args.no_failure_images:
                        base_name = "frame_{:06d}".format(processed_frames)
                        cv2.imwrite(os.path.join(failure_dir, base_name + "_overlay.png"), overlay)
                        cv2.imwrite(os.path.join(failure_dir, base_name + "_debug.png"), debug_grid)
                        cv2.imwrite(os.path.join(failure_dir, base_name + "_edge.png"), result["edge"])
                        cv2.imwrite(os.path.join(failure_dir, base_name + "_roi_edge.png"), result["roi_edge"])

                max_fail_count = max(max_fail_count, int(result["fail_count"]))
                max_abs_center_jump = max(max_abs_center_jump, abs(int(result["center_jump"])))
                if result["lane_width_jump"] is not None:
                    max_abs_lane_width_jump = max(max_abs_lane_width_jump, abs(float(result["lane_width_jump"])))

                processed_frames += 1

                if processed_frames % 100 == 0:
                    print("[INFO] processed={} mode={} fail={} roi_mean={:.1f} edge={} hough={}".format(
                        processed_frames,
                        result["mode"],
                        result["fail_count"],
                        result["roi_mean"],
                        result["edge_pixel_count"],
                        result["hough_line_count"],
                    ))

    if overlay_writer is not None:
        overlay_writer.release()
    if debug_writer is not None:
        debug_writer.release()

    summary = {
        "bag": os.path.abspath(args.bag),
        "topic": args.topic,
        "output_dir": os.path.abspath(args.output_dir),
        "total_messages_seen_on_topic": total_frames,
        "processed_frames": processed_frames,
        "every_n": args.every_n,
        "max_frames_arg": args.max_frames,
        "mode_counts": mode_counts,
        "suspicious_frames": suspicious_frames,
        "max_fail_count": max_fail_count,
        "max_abs_center_jump": max_abs_center_jump,
        "max_abs_lane_width_jump": max_abs_lane_width_jump,
        "config": asdict(cfg),
        "outputs": {
            "csv": os.path.abspath(csv_path),
            "summary_json": os.path.abspath(summary_path),
            "overlay_video": os.path.abspath(overlay_video_path) if not args.no_video else None,
            "debug_video": os.path.abspath(debug_video_path) if not args.no_video else None,
            "failure_frames_dir": os.path.abspath(failure_dir) if not args.no_failure_images else None,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[DONE] 분석 완료")
    print("[DONE] processed_frames      : {}".format(processed_frames))
    print("[DONE] mode_counts           : {}".format(mode_counts))
    print("[DONE] suspicious_frames     : {}".format(suspicious_frames))
    print("[DONE] max_fail_count        : {}".format(max_fail_count))
    print("[DONE] csv                   : {}".format(csv_path))
    print("[DONE] summary               : {}".format(summary_path))
    if not args.no_video:
        print("[DONE] overlay video         : {}".format(overlay_video_path))
        print("[DONE] debug grid video      : {}".format(debug_video_path))
    if not args.no_failure_images:
        print("[DONE] failure frames dir    : {}".format(failure_dir))


def main() -> None:
    args = parse_args()
    analyze_bag(args)


if __name__ == "__main__":
    main()
