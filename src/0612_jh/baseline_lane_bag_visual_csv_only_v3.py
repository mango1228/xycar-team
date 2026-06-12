#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_lane_bag_visual_csv_only_v3.py

목적:
    rosbag에 저장된 카메라 raw image 토픽을 offline으로 읽어서,
    현재 기본 차선 검출 알고리즘(BGR -> Gray -> Blur -> Canny -> ROI -> HoughLinesP)을
    프레임별로 검증한다.

출력:
    - CSV 1개만 저장
    - mp4 저장 없음
    - png/frame image 저장 없음
    - 실행 중 OpenCV 창으로 시각화
    - 화면/터미널/CSV에 rel_time 표시

기본 실행 예시:
    source /opt/ros/noetic/setup.bash

    python3 baseline_lane_bag_visual_csv_only.py \
      --bag /home/kim/smo_term/xycar-team/src/my_motor/src/2026-06-08-15-10-10.bag \
      --output-dir /home/kim/smo_term/xycar-team/src/my_motor/src/lane_bag_analysis_visual_csv \
      --topic /usb_cam/image_raw \
      --view both

키 조작:
    q 또는 ESC : 종료
    SPACE      : 일시정지/재개
    n          : 일시정지 상태에서 다음 프레임 1개 진행

주의:
    - GUI가 가능한 Ubuntu 환경에서 실행해야 cv2.imshow 창이 뜬다.
    - SSH/headless 환경이면 --view none 으로 CSV만 저장하거나, X11 forwarding/VNC가 필요하다.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import rosbag
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CompressedImage
except Exception as exc:
    print("[ERROR] ROS 관련 모듈 import 실패")
    print("        먼저 ROS 환경을 source 하세요.")
    print("        예: source /opt/ros/noetic/setup.bash")
    print("        원인: {}".format(exc))
    sys.exit(1)


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
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)

        y0 = max(0, int(cfg.offset))
        y1 = min(edge.shape[0], int(cfg.offset + cfg.gap))
        x0 = 0
        x1 = min(edge.shape[1], int(cfg.width))

        roi_gray = gray[y0:y1, x0:x1]
        roi_edge = edge[y0:y1, x0:x1]

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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_bgr_from_msg(bridge: CvBridge, msg: Any) -> np.ndarray:
    """
    rosbag에서 읽은 이미지 메시지를 OpenCV BGR 이미지로 변환한다.

    주의점:
      - rosbag 메시지는 workspace/source 상태에 따라 isinstance(msg, Image)가 예상대로 안 맞는 경우가 있다.
      - 그래서 msg._type 문자열을 우선 사용한다.
      - usb_cam raw가 yuv422/yuyv로 저장된 경우 cv_bridge의 bgr8 변환이 실패할 수 있어 직접 변환을 추가한다.
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

        # 일반적인 경우: cv_bridge가 바로 bgr8로 변환
        if encoding in ["bgr8", "rgb8", "mono8", "8uc1", "8uc3"]:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return img

        # usb_cam에서 흔한 YUYV/YUV422 계열.
        # sensor_msgs/Image에서는 height, width, step, data를 직접 사용해야 padding에 덜 취약하다.
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

        # 마지막 fallback: cv_bridge passthrough 후 채널 수에 따라 BGR로 정리
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            # encoding이 rgb8이면 BGR로 변환, 그 외 3채널은 이미 BGR에 가깝다고 보고 사용
            if encoding == "rgb8":
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        raise RuntimeError("지원하지 않는 Image encoding/shape: encoding={} shape={}".format(encoding, getattr(img, "shape", None)))

    raise TypeError("지원하지 않는 메시지 타입: msg._type={} python_type={}".format(msg_type, type(msg)))


def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.55, thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, result: Dict[str, Any], cfg: BaselineConfig) -> np.ndarray:
    overlay = frame_bgr.copy()

    h, w = overlay.shape[:2]
    y0 = max(0, int(cfg.offset))
    y1 = min(h, int(cfg.offset + cfg.gap))

    cv2.rectangle(overlay, (0, y0), (min(cfg.width, w) - 1, y1 - 1), (255, 0, 0), 2)
    cv2.line(overlay, (cfg.width // 2, 0), (cfg.width // 2, h - 1), (120, 120, 120), 1)

    # Hough 전체 후보: ROI 좌표 -> 원본 좌표로 변환해서 그림
    for x1, y1_l, x2, y2_l in result["all_lines"]:
        cv2.line(overlay, (x1, y1_l + y0), (x2, y2_l + y0), (0, 255, 255), 1)

    # 최종 선택 차선
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

    rel_time = result.get("rel_time", None)
    rel_time_text = "rel={:.3f}s".format(rel_time) if rel_time is not None else "rel=N/A"

    draw_text(overlay, "{}  frame={}  mode={}  center={}  err={}".format(
        rel_time_text, result.get("frame_idx", ""), result["mode"], result["center"], result["center_error"]), (10, 25))
    draw_text(overlay, "roi_mean={:.1f}  edge={}  hough={}  L/R={}/{}  fail={}".format(
        result["roi_mean"], result["edge_pixel_count"], result["hough_line_count"],
        result["left_line_count"], result["right_line_count"], result["fail_count"]), (10, 50))
    draw_text(overlay, "lpos={}  rpos={}  width={}  corner={}".format(
        result["lpos"], result["rpos"], result["lane_width"], result["corner"]), (10, 75))

    if result["suspicious_reasons"]:
        draw_text(overlay, "WARN: {}".format(result["suspicious_reasons"]), (10, 100))

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


def make_debug_grid(frame_bgr: np.ndarray, overlay: np.ndarray, result: Dict[str, Any], cfg: BaselineConfig) -> np.ndarray:
    panel_w, panel_h = 320, 240

    gray_bgr = cv2.cvtColor(result["gray"], cv2.COLOR_GRAY2BGR)
    edge_bgr = cv2.cvtColor(result["edge"], cv2.COLOR_GRAY2BGR)
    roi_gray_bgr = cv2.cvtColor(result["roi_gray"], cv2.COLOR_GRAY2BGR)
    roi_edge_bgr = cv2.cvtColor(result["roi_edge"], cv2.COLOR_GRAY2BGR)

    roi_hough = cv2.cvtColor(result["roi_edge"], cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in result["all_lines"]:
        cv2.line(roi_hough, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for x1, y1, x2, y2 in result["selected_left"]:
        cv2.line(roi_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x1, y1, x2, y2 in result["selected_right"]:
        cv2.line(roi_hough, (x1, y1), (x2, y2), (255, 0, 255), 2)

    p1 = make_panel(frame_bgr, "Original", (panel_w, panel_h))
    p2 = make_panel(gray_bgr, "Gray", (panel_w, panel_h))
    p3 = make_panel(edge_bgr, "Canny edge", (panel_w, panel_h))
    p4 = make_panel(roi_gray_bgr, "ROI gray", (panel_w, panel_h))
    p5 = make_panel(roi_hough, "ROI Hough", (panel_w, panel_h))
    p6 = make_panel(overlay, "Final overlay", (panel_w, panel_h))

    top = np.hstack([p1, p2, p3])
    bottom = np.hstack([p4, p5, p6])
    grid = np.vstack([top, bottom])
    return grid


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def open_csv_writer(csv_path: str) -> Tuple[Any, csv.DictWriter]:
    f = open(csv_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "frame_idx",
        "processed_idx",
        "timestamp",
        "rel_time",
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
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rosbag camera raw 기반 기본 차선 검출 CSV 저장 + 실시간 시각화 전용 analyzer"
    )

    parser.add_argument("--bag", required=True, help="입력 rosbag 파일 경로")
    parser.add_argument("--topic", default="/usb_cam/image_raw", help="카메라 이미지 토픽")
    parser.add_argument("--output-dir", required=True, help="CSV 저장 폴더")
    parser.add_argument("--csv-name", default="lane_detection_log.csv", help="저장할 CSV 파일명")

    parser.add_argument("--view", choices=["overlay", "debug", "both", "none"], default="overlay",
                        help="실행 중 화면 시각화 방식. mp4 저장은 하지 않음")
    parser.add_argument("--display-scale", type=float, default=1.0, help="표시 창 크기 배율")
    parser.add_argument("--delay-ms", type=int, default=1, help="cv2.waitKey 대기 시간")
    parser.add_argument("--every-n", type=int, default=1, help="N프레임마다 1개 처리")
    parser.add_argument("--max-frames", type=int, default=0, help="최대 처리 프레임 수. 0이면 전체")

    # 기존 baseline 파라미터를 실행 시 조정할 수 있게 둠. 기본값은 현재 코드 기준.
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.bag):
        print("[ERROR] bag 파일이 없습니다: {}".format(args.bag))
        sys.exit(1)

    if args.every_n < 1:
        print("[ERROR] --every-n은 1 이상이어야 합니다.")
        sys.exit(1)

    ensure_dir(args.output_dir)
    csv_path = os.path.join(args.output_dir, args.csv_name)

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
    )

    detector = BaselineLaneDetector(cfg)
    bridge = CvBridge()

    total_msg_count = 0
    processed_count = 0
    first_stamp_sec: Optional[float] = None
    mode_count: Dict[str, int] = {"BOTH": 0, "LEFT": 0, "RIGHT": 0, "NONE": 0}
    paused = False
    step_once = False

    print("[INFO] 입력 bag     : {}".format(args.bag))
    print("[INFO] 이미지 토픽  : {}".format(args.topic))
    print("[INFO] CSV 저장     : {}".format(csv_path))
    print("[INFO] 화면 표시    : {}".format(args.view))
    print("[INFO] mp4/png 저장 : 하지 않음")
    print("[INFO] 종료 키      : q 또는 ESC")
    print("[INFO] 일시정지      : SPACE")
    print("[INFO] 정지 후 1프레임 이동: n")
    print("[INFO] 시간 기준     : /usb_cam/image_raw 첫 프레임 기준 rel_time")

    csv_file, writer = open_csv_writer(csv_path)

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

            decode_fail_count = 0

            for topic, msg, stamp in bag.read_messages(topics=[args.topic]):
                if total_msg_count % args.every_n != 0:
                    total_msg_count += 1
                    continue

                if args.max_frames > 0 and processed_count >= args.max_frames:
                    break

                try:
                    frame_bgr = to_bgr_from_msg(bridge, msg)
                except Exception as exc:
                    decode_fail_count += 1
                    if decode_fail_count <= 10 or decode_fail_count % 100 == 0:
                        print("[WARN] frame decode 실패: frame_idx={} | msg_type={} | encoding={} | {}".format(
                            total_msg_count,
                            getattr(msg, "_type", ""),
                            getattr(msg, "encoding", ""),
                            exc,
                        ))
                    total_msg_count += 1
                    continue

                stamp_sec = float(stamp.to_sec())
                if first_stamp_sec is None:
                    first_stamp_sec = stamp_sec
                rel_time = stamp_sec - first_stamp_sec

                result = detector.process(frame_bgr)
                result["rel_time"] = rel_time
                result["frame_idx"] = total_msg_count
                result["processed_idx"] = processed_count
                mode_count[result["mode"]] = mode_count.get(result["mode"], 0) + 1

                row = {
                    "frame_idx": total_msg_count,
                    "processed_idx": processed_count,
                    "timestamp": "{:.9f}".format(stamp_sec),
                    "rel_time": "{:.6f}".format(rel_time),
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
                writer.writerow(row)

                if processed_count % 50 == 0:
                    csv_file.flush()
                    print("[INFO] rel_time={:.3f}s | processed={} | frame_idx={} | mode={} | center={} | roi_mean={:.1f} | hough={}".format(
                        rel_time,
                        processed_count,
                        total_msg_count,
                        result["mode"],
                        result["center"],
                        result["roi_mean"],
                        result["hough_line_count"],
                    ))

                if args.view != "none":
                    overlay = draw_overlay(frame_bgr, result, cfg)

                    if args.view in ["overlay", "both"]:
                        cv2.imshow("baseline overlay - no mp4 save", resize_for_display(overlay, args.display_scale))

                    if args.view in ["debug", "both"]:
                        debug_grid = make_debug_grid(frame_bgr, overlay, result, cfg)
                        cv2.imshow("baseline debug grid - no mp4 save", resize_for_display(debug_grid, args.display_scale))

                    while True:
                        key = cv2.waitKey(args.delay_ms) & 0xFF

                        if key in [ord("q"), 27]:
                            print("[INFO] 사용자 종료")
                            raise KeyboardInterrupt

                        if key == ord(" "):
                            paused = not paused
                            print("[INFO] paused={}".format(paused))

                        if paused:
                            key2 = cv2.waitKey(0) & 0xFF
                            if key2 in [ord("q"), 27]:
                                print("[INFO] 사용자 종료")
                                raise KeyboardInterrupt
                            if key2 == ord(" "):
                                paused = False
                                print("[INFO] paused=False")
                                break
                            if key2 == ord("n"):
                                step_once = True
                                break
                            continue

                        break

                    if step_once:
                        paused = True
                        step_once = False

                processed_count += 1
                total_msg_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        csv_file.flush()
        csv_file.close()
        if args.view != "none":
            cv2.destroyAllWindows()

    print("\n========== 분석 종료 ==========")
    print("전체 읽은 frame index 기준 마지막 값 : {}".format(total_msg_count))
    print("처리한 프레임 수                 : {}".format(processed_count))
    print("CSV 저장                         : {}".format(csv_path))
    print("mode count                       : {}".format(mode_count))
    print("mp4/png 저장                     : 없음")


if __name__ == "__main__":
    main()
