#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import time
import cv2
import numpy as np
from std_msgs.msg import Int8
from config import Config
from lane_drive import ImageProcessor, LidarProcessor, XycarDriver

class XycarController:
    # 주행 상태
    STATE_DRIVE          = "DRIVE"
    STATE_CROSSWALK_STOP = "CROSSWALK_STOP"   # 횡단보도 정지(검증). 10초 유지=진짜 / 사라지면 오탐 복귀
    STATE_HATCH_VERIFY   = "HATCH_VERIFY"     # 빗금 정지 검증. 1초 유지=진짜 / 사라지면 오탐 복귀
    STATE_HATCH_ADVANCE  = "HATCH_ADVANCE"    # 빗금 확정 후 1초 전진
    STATE_HATCH_STOP     = "HATCH_STOP"       # 영구 정지

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

        # 특수구역 상태머신 상태 (컨트롤러 소유)
        self.drive_state         = self.STATE_DRIVE
        self.stop_start_time     = None         # 횡단보도 정지 시작 시각
        self.hatch_verify_start  = None         # 빗금 검증 시작 시각
        self.advance_start       = None         # 빗금 전진 시작 시각
        self.resume_grace_until  = 0.0          # 횡단보도 성공 후 둘 다 무시하는 유예 마감 시각
        self.last_cross_seen     = 0.0          # 횡단보도가 마지막으로 검출된 시각
        self.last_hatch_seen     = 0.0          # 빗금이 마지막으로 검출된 시각
        self.cross_now           = False        # 최신 메시지의 횡단보도 검출값
        self.hatch_now           = False        # 최신 메시지의 빗금 검출값
        self.cross_consecutive   = 0
        self.hatch_consecutive   = 0
        self.hatch_miss          = 0            # 빗금 연속 미검출 카운트 (miss tolerance용)
        self.node_start_time     = None         # run()에서 워밍업 후 설정

        # /special_zone 구독: 검출은 별도 노드(special_zone_detector)가 담당.
        # 콜백은 최신값만 저장(가벼움), 카운터 갱신은 메인 루프가 메시지 단위로 처리.
        self.sz_raw = 0       # 비트마스크: 1=횡단보도, 2=빗금
        self.sz_new = False   # 새 검출 메시지 도착 플래그
        rospy.Subscriber('/special_zone', Int8, self.special_zone_cb, queue_size=1)

        rospy.on_shutdown(self.shutdown)  # 안전 정지 (콜백 등록: 괄호 없이 함수 참조)

        self.rate = rospy.Rate(30)

    def special_zone_cb(self, msg):
        # 가벼운 콜백: 최신 검출 비트마스크만 저장 + 새 메시지 플래그
        self.sz_raw = msg.data
        self.sz_new = True

    def compute_pid_angle(self, center):
        """center -> PID 조향각. 적분 와인드업 클램프 포함."""
        now   = time.time()
        error = center - self.cfg.width // 2

        if self.pid_time is None:
            # 첫 프레임/reset 직후: 미분 기준이 없음 → D 스킵, P만 (조향 튐 방지)
            self.pid_time   = now
            self.prev_error = error
            return max(-50, min(50, error * self.cfg.gain))

        dt = max(now - self.pid_time, 1e-6)
        self.pid_time = now

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

            if not self.cfg.enable_special_zone:
                # ===== 기존 차선주행 경로 (특수구역 off) =====
                angle = self.compute_pid_angle(center)
                self.xycar_driver.drive(angle, self.cfg.speed)
            else:
                # ===== 특수구역 정지 경로 =====
                # 검출은 별도 노드가 /special_zone 으로 보냄. 메시지 단위로 카운터/최근검출시각 갱신.
                if self.sz_new:
                    self.sz_new = False
                    self.cross_now = bool(self.sz_raw & 1)
                    self.hatch_now = bool(self.sz_raw & 2)
                    if self.cross_now:
                        self.last_cross_seen = now
                    if self.hatch_now:
                        self.last_hatch_seen = now

                    # 트리거 카운터는 DRIVE 상태에서만 갱신.
                    # (정지 중 누적되면 재출발 직후 무메시지 프레임에서 또 트리거되는 버그 방지)
                    if self.drive_state == self.STATE_DRIVE:
                        # 유예: 시작 직후(startup_grace) 또는 횡단보도 성공 후(resume_grace) → 둘 다 무시
                        in_grace = ((now - self.node_start_time) < self.cfg.startup_grace_sec
                                    or now < self.resume_grace_until)
                        if self.cross_now and not in_grace:
                            self.cross_consecutive += 1
                            self.hatch_consecutive  = 0
                            self.hatch_miss         = 0
                        else:
                            self.cross_consecutive = 0
                            if self.hatch_now and not in_grace:
                                self.hatch_consecutive += 1
                                self.hatch_miss = 0
                            else:
                                # 연속 미검출이 tolerance를 넘어야만 카운터 리셋(깜빡임 흡수)
                                self.hatch_miss += 1
                                if self.hatch_miss >= self.cfg.hatch_miss_tolerance:
                                    self.hatch_consecutive = 0

                # 상태머신은 매 프레임(30Hz) 동작
                if self.drive_state == self.STATE_HATCH_STOP:
                    self.xycar_driver.drive(0, 0)                       # 영구 정지

                elif self.drive_state == self.STATE_HATCH_ADVANCE:
                    # 빗금 확정 후 차선 따라 전진 → 시간 다 되면 영구 정지
                    if now - self.advance_start >= self.cfg.hatch_advance_sec:
                        rospy.loginfo("[special_zone] 빗금 전진 완료 -> 영구 정지")
                        self.drive_state = self.STATE_HATCH_STOP
                        self.xycar_driver.drive(0, 0)
                    else:
                        angle = self.compute_pid_angle(center)
                        self.xycar_driver.drive(angle, self.cfg.speed)

                elif self.drive_state == self.STATE_CROSSWALK_STOP:
                    # 정지 중 검증: 10초 유지=진짜 / 사라지면 오탐 즉시 복귀
                    self.xycar_driver.drive(0, 0)
                    if now - self.last_cross_seen > self.cfg.cross_lost_sec:
                        rospy.loginfo("[special_zone] 횡단보도 아님(오탐) -> 재출발")
                        self.reset_pid()
                        self.drive_state = self.STATE_DRIVE
                    elif now - self.stop_start_time >= self.cfg.crosswalk_stop_sec:
                        rospy.loginfo("[special_zone] 횡단보도 10초 정지 완료 -> 재출발(+%.1f초 유예)" % self.cfg.post_resume_grace_sec)
                        self.resume_grace_until = now + self.cfg.post_resume_grace_sec
                        self.reset_pid()
                        self.drive_state = self.STATE_DRIVE

                else:  # STATE_DRIVE
                    if self.cross_consecutive >= self.cfg.cross_confirm_frames:
                        rospy.loginfo("[special_zone] 횡단보도 감지 -> 정지(검증)")
                        self.drive_state       = self.STATE_CROSSWALK_STOP
                        self.stop_start_time   = now
                        self.last_cross_seen   = now
                        self.cross_consecutive = 0   # 트리거 후 리셋 (재출발 직후 재트리거 방지)
                        self.hatch_consecutive = 0
                        self.hatch_miss        = 0
                        self.reset_pid()
                        self.xycar_driver.drive(0, 0)

                    elif self.hatch_consecutive >= self.cfg.hatch_confirm_frames:
                        rospy.loginfo("[special_zone] 빗금 감지 -> 차선 따라 %.1f초 전진 후 영구정지" % self.cfg.hatch_advance_sec)
                        self.drive_state       = self.STATE_HATCH_ADVANCE
                        self.advance_start     = now
                        self.cross_consecutive = 0
                        self.hatch_consecutive = 0
                        self.hatch_miss        = 0
                        # PID 상태 유지(계속 주행) → 멈춤 없이 부드럽게 전진

                    else:
                        angle = self.compute_pid_angle(center)
                        self.xycar_driver.drive(angle, self.cfg.speed)

            if self.show_debug:
                # 특수구역 상태/카운트다운 (검출 오버레이는 detector 노드 자체 창)
                if self.cfg.enable_special_zone:
                    self._draw_status(frame, now)
                self.image_processor.draw_debug(frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos, corner, left_slope)

            count += 1
            if count % 30 == 0:
                half_str = "%.0f" % self.image_processor.ema_half_width if self.image_processor.ema_half_width is not None else "-"
                print("state=%s mode=%s center=%d ema_half=%s"
                    % (self.drive_state, mode, center, half_str))

            self.rate.sleep()

    def _draw_status(self, frame, now):
        """특수구역 정지 상태/남은 시간을 차선 디버그 창에 크게 표시."""
        if self.drive_state == self.STATE_CROSSWALK_STOP:
            remain = max(0.0, self.cfg.crosswalk_stop_sec - (now - self.stop_start_time))
            cv2.putText(frame, "CROSSWALK STOP  %.1fs" % remain, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        elif self.drive_state == self.STATE_HATCH_ADVANCE:
            remain = max(0.0, self.cfg.hatch_advance_sec - (now - self.advance_start))
            cv2.putText(frame, "HATCH ADVANCE  %.1fs" % remain, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif self.drive_state == self.STATE_HATCH_STOP:
            cv2.putText(frame, "HATCH STOP", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    def shutdown(self):
        self.xycar_driver.shutdown()
        self.image_processor.shutdown()

if __name__ == '__main__':
    xycar_controller = XycarController()
    xycar_controller.run()
