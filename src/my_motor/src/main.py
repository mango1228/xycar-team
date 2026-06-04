#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import time
import numpy as np
from config import Config
from lane_drive import ImageProcessor, LidarProcessor, XycarDriver
from special_zone import SpecialZoneProcessor

class XycarController:
    # 주행 상태
    STATE_DRIVE          = "DRIVE"
    STATE_CROSSWALK_STOP = "CROSSWALK_STOP"
    STATE_HATCH_STOP     = "HATCH_STOP"

    def __init__(self):
        rospy.init_node('lane_drive')

        self.cfg = Config()

        self.show_debug = self.cfg.show_debug

        if self.show_debug and not os.environ.get('DISPLAY'):
            rospy.logwarn("DISPLAY 없음 - 디버그 창 비활성화")
            self.show_debug = False

        self.prev_error = 0.0
        self.i_error = 0.0
        self.pid_time = None

        self.image_processor = ImageProcessor(self.cfg)
        self.lidar_processor = LidarProcessor(self.cfg)
        self.xycar_driver = XycarDriver()
        self.special_zone_processor = SpecialZoneProcessor(self.cfg)

        # 특수구역 상태머신 상태 (컨트롤러 소유)
        self.drive_state               = self.STATE_DRIVE
        self.stop_start_time           = None
        self.cross_cooldown_until      = 0.0
        self.resume_grace_until        = 0.0    # 횡단보도 재출발 직후 감지 유예 마감 시각
        self.hatch_first_detected_time = None
        self.cross_consecutive         = 0
        self.hatch_consecutive         = 0
        self.hatch_miss                = 0      # 빗금 연속 미검출 카운트 (miss tolerance용)
        self.node_start_time           = None   # run()에서 워밍업 후 설정

        # 검사 결과 캐시 (detect_every>1 일 때 검사 안 한 프레임에서 오버레이용으로 재사용)
        self.last_cross_blocks = []
        self.last_hatch_rects  = []
        self.last_hatch_pct    = 0.0

        rospy.on_shutdown(self.shutdown)  # 안전 정지 (콜백 등록: 괄호 없이 함수 참조)

        self.rate = rospy.Rate(30)

    def compute_pid_angle(self, center):
        """center -> PID 조향각. 적분 와인드업 클램프 포함."""
        now = time.time()
        dt  = (now - self.pid_time) if self.pid_time is not None else 1e-6
        dt  = max(dt, 1e-6)
        self.pid_time = now

        error = center - self.cfg.width // 2
        self.i_error += error * dt
        self.i_error = float(np.clip(self.i_error, -self.cfg.i_clamp, self.cfg.i_clamp))  # 와인드업 방지
        d_out = (error - self.prev_error) / dt
        self.prev_error = error

        angle = error * self.cfg.gain + self.i_error * self.cfg.gain_i + d_out * self.cfg.gain_d
        return max(-50, min(50, angle))

    def reset_pid(self):
        """정지/재출발 시 PID 상태 초기화 (D항 스파이크 / I항 누적 방지)."""
        self.prev_error = 0.0
        self.i_error    = 0.0
        self.pid_time   = None

    def run(self):
        rospy.sleep(2.0)  # 센서 워밍업 대기
        self.node_start_time = time.time()
        print("lane_drive started")

        count = 0
        while not rospy.is_shutdown():
            image = self.image_processor.image
            if image.size == 0:
                self.rate.sleep()
                continue

            frame = image.copy()
            center, mode, left, right, lpos, rpos, corner, left_slope = self.image_processor.process(frame)
            cam_center = center

            roi_data = self.lidar_processor.get_roi_data(self.lidar_processor.lidar_scan)
            gaps     = self.lidar_processor.get_gaps(roi_data)

            bisector = None
            lidar_c  = None
            if gaps:
                largest  = max(gaps, key=lambda g: g[1] - g[0])
                bisector = (largest[0] + largest[1]) / 2.0
                # ROI_ANG_CENTER - bisector: laser_frame에서 car left/right 방향 보정
                # bisector > CENTER(=-90°) → car's left → negative offset → steer left
                dev = self.lidar_processor.roi_ang_center - bisector
                lidar_c = max(0, min(self.cfg.width - 1, int(self.cfg.width // 2 + dev * self.cfg.lidar_center_gain)))

            self.lidar_processor.publish_roi_markers(roi_data, gaps, bisector, self.lidar_processor.lidar_scan)

            if lidar_c is not None:
                # 차선 경계 안으로 클램프 (차선 바깥 조향 방지)
                if lpos is not None:
                    lidar_c = max(lidar_c, lpos)
                if rpos is not None:
                    lidar_c = min(lidar_c, rpos)
                if abs(lidar_c - self.cfg.width // 2) > abs(center - self.cfg.width // 2):
                    center = lidar_c
                    mode = mode + "+LIDAR"

            now = time.time()
            stop_remain        = 0.0
            hatch_delay_remain = 0.0

            if not self.cfg.enable_special_zone:
                # ===== 기존 차선주행 경로 (특수구역 off) =====
                angle = self.compute_pid_angle(center)
                self.xycar_driver.drive(angle, self.cfg.speed)
            else:
                # ===== 특수구역 정지 경로 =====
                # 검사 호출 + confirm 카운터 증가는 detect_every 주기에만 (디바운스 의미 보존)
                if count % self.cfg.detect_every == 0:
                    cross_detected, self.last_cross_blocks, _ = \
                        self.special_zone_processor.detect_crosswalk(frame)
                    hatch_detected, self.last_hatch_rects, self.last_hatch_pct = \
                        self.special_zone_processor.detect_hatch(frame)

                    # 유예: 시작 직후(startup_grace) 또는 횡단보도 재출발 직후(resume_grace)
                    in_grace = ((now - self.node_start_time) < self.cfg.startup_grace_sec
                                or now < self.resume_grace_until)
                    cross_active = cross_detected and now > self.cross_cooldown_until and not in_grace
                    if cross_active:
                        self.cross_consecutive += 1
                        self.hatch_consecutive  = 0
                        self.hatch_miss         = 0
                    else:
                        self.cross_consecutive = 0
                        if hatch_detected and not in_grace:
                            self.hatch_consecutive += 1
                            self.hatch_miss = 0
                        else:
                            # 연속 미검출이 tolerance를 넘어야만 카운터 리셋(깜빡임 흡수)
                            self.hatch_miss += 1
                            if self.hatch_miss >= self.cfg.hatch_miss_tolerance:
                                self.hatch_consecutive = 0

                # 상태 실행/주행/정지타이머/하치딜레이는 캐시 카운터로 매 프레임(30Hz) 동작
                if self.drive_state == self.STATE_HATCH_STOP:
                    self.xycar_driver.drive(0, 0)

                elif self.drive_state == self.STATE_CROSSWALK_STOP:
                    self.xycar_driver.drive(0, 0)
                    elapsed     = now - self.stop_start_time
                    stop_remain = max(0.0, self.cfg.crosswalk_stop_sec - elapsed)
                    if elapsed >= self.cfg.crosswalk_stop_sec:
                        rospy.loginfo("[special_zone] 횡단보도 정지 완료 -> 재출발")
                        self.cross_cooldown_until = now + self.cfg.cross_cooldown_sec
                        # 재출발 직후 유예: 차가 횡단보도를 지나기 전 빗금 오발동(영구정지) 방지
                        self.resume_grace_until = now + self.cfg.post_resume_grace_sec
                        self.reset_pid()
                        self.drive_state = self.STATE_DRIVE

                else:  # STATE_DRIVE
                    if self.cross_consecutive >= self.cfg.cross_confirm_frames:
                        rospy.loginfo("[special_zone] 횡단보도 감지 -> %.1f초 정지" % self.cfg.crosswalk_stop_sec)
                        self.drive_state               = self.STATE_CROSSWALK_STOP
                        self.stop_start_time           = now
                        self.hatch_first_detected_time = None
                        self.reset_pid()
                        self.xycar_driver.drive(0, 0)

                    elif self.hatch_consecutive >= self.cfg.hatch_confirm_frames:
                        if self.hatch_first_detected_time is None:
                            self.hatch_first_detected_time = now
                            rospy.loginfo("[special_zone] 빗금 확정 -> %.1f초 후 정지" % self.cfg.hatch_delay_sec)
                        elapsed_hatch = now - self.hatch_first_detected_time
                        if elapsed_hatch >= self.cfg.hatch_delay_sec:
                            rospy.loginfo("[special_zone] 빗금 구역 -> 영구 정지")
                            self.drive_state = self.STATE_HATCH_STOP
                            self.reset_pid()
                            self.xycar_driver.drive(0, 0)
                        else:
                            hatch_delay_remain = self.cfg.hatch_delay_sec - elapsed_hatch
                            angle = self.compute_pid_angle(center)
                            self.xycar_driver.drive(angle, self.cfg.speed)

                    else:
                        self.hatch_first_detected_time = None
                        angle = self.compute_pid_angle(center)
                        self.xycar_driver.drive(angle, self.cfg.speed)

            if self.show_debug:
                # 특수구역 오버레이를 먼저 그리고, draw_debug가 마지막에 한 번만 창 flush
                if self.cfg.enable_special_zone:
                    self.special_zone_processor.draw_overlay(
                        frame, self.last_cross_blocks, self.last_hatch_rects,
                        self.drive_state, stop_remain, self.last_hatch_pct, hatch_delay_remain)
                self.image_processor.draw_debug(frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos, corner, left_slope)

            count += 1
            if count % 30 == 0:
                half_str = "%.0f" % self.image_processor.ema_half_width if self.image_processor.ema_half_width is not None else "-"
                print("state=%s mode=%s center=%d ema_half=%s"
                    % (self.drive_state, mode, center, half_str))

            self.rate.sleep()

    def shutdown(self):
        self.xycar_driver.shutdown()
        self.image_processor.shutdown()

if __name__ == '__main__':
    xycar_controller = XycarController()
    xycar_controller.run()
