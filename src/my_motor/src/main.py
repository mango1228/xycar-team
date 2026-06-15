#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import time
from std_msgs.msg import Int8
from config import Config
from lane_drive import ImageProcessor, LidarProcessor, XycarDriver, LaneFollower, ARtagDetector
from controller import PIDController, SpecialZoneController

class XycarController:
    # # 주행 상태
    # STATE_DRIVE          = "DRIVE"
    # STATE_CROSSWALK_STOP = "CROSSWALK_STOP"   # 횡단보도 정지(검증). 10초 유지=진짜 / 사라지면 오탐 복귀
    # STATE_HATCH_VERIFY   = "HATCH_VERIFY"     # 빗금 정지 검증. 1초 유지=진짜 / 사라지면 오탐 복귀
    # STATE_HATCH_ADVANCE  = "HATCH_ADVANCE"    # 빗금 확정 후 1초 전진
    # STATE_HATCH_STOP     = "HATCH_STOP"       # 영구 정지

    def __init__(self):
        rospy.init_node('lane_drive')

        self.cfg = Config()

        self.show_debug = self.cfg.show_debug
        self.ar_follow = False
        self.ar_prev = False        # 직전 프레임에 AR이 보였는지
        self.ar_hold_until = 0.0    # 사라진 뒤 이 시각까지 유지
        self.ar_hold_sec = 5.0      # 사라진 후 유지할 시간(초)

        if self.show_debug and not os.environ.get('DISPLAY'):
            rospy.logwarn("DISPLAY 없음 - 디버그 창 비활성화")
            self.show_debug = False

        self.pid_controller = PIDController(self.cfg)
        self.image_processor = ImageProcessor(self.cfg)
        self.lidar_processor = LidarProcessor(self.cfg)
        self.xycar_driver = XycarDriver()
        self.lane_follower = LaneFollower(self.cfg)
        self.special_zone_controller = SpecialZoneController(self.cfg, self.pid_controller)
        self.ar_tag_detector = ARtagDetector()


        rospy.on_shutdown(self.shutdown)  # 안전 정지 (콜백 등록: 괄호 없이 함수 참조)

        self.rate = rospy.Rate(30)

    def run(self):
        rospy.sleep(2.0)  # 센서 워밍업 대기
        self.special_zone_controller.node_start_time = time.time()
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

            lidar_c = None
            bisector = None

            if gaps:
                lidar_c, bisector = self.lane_follower.correct_lane(gaps, self.lidar_processor.roi_ang_center)
                
            self.lidar_processor.publish_roi_markers(roi_data, gaps, bisector, self.lidar_processor.lidar_scan)
            
            # if self.ar_tag_detector.ar_detected and lidar_c is not None:
            #     center = lidar_c
            #     mode = "LIDAR_ONLY"
            #     self.ar_follow = True
            #     print("AR detected")
            # else:
            #     if self.ar_follow == True:
            #         center = lidar_c
            #         mode = "LIDAR_ONLY"
            #         self.ar_follow = False
            #         now = time.time()
            #     if now <= self.ar_hold_time and lidar_c is not None:
            #         center = lidar_c
            #         mode = "LIDAR_ONLY"
            #     else: #lidar_c is not None:
            #         # 차선 경계 안으로 클램프 (차선 바깥 조향 방지)
            #         if lpos is not None:
            #             lidar_c = max(lidar_c, lpos)
            #         if rpos is not None:
            #             lidar_c = min(lidar_c, rpos)
            #         if abs(lidar_c - self.cfg.width // 2) > abs(center - self.cfg.width // 2):
            #             center = lidar_c
            #             mode = mode + "+LIDAR"
            now = time.time()
            ar_now = self.ar_tag_detector.ar_detected and lidar_c is not None

            if ar_now and self.ar_tag_detector.get_distance():
                # AR이 보이는 동안: 라이다 모드 (타이머는 아직 시작 안 함)
                center = lidar_c
                mode = "LIDAR_ONLY"
                print("AR detected")

            if self.ar_prev and not ar_now:
                # ★ 보였다 → 사라진 바로 그 순간: 여기서 N초 타이머 시작
                self.ar_hold_until = now + self.ar_hold_sec
                if lidar_c is not None:
                    center = lidar_c
                mode = "LIDAR_HOLD"
                print("Now follow LIDAR")
            
            if now < self.ar_hold_until:
                # 사라진 뒤 유지 시간 안: 계속 라이다 모드
                if lidar_c is not None:
                    center = lidar_c
                mode = "LIDAR_HOLD"
            else:
                # 유지 시간 끝: 평소 주행 + 라이다 보정
                if lidar_c is not None:
                    if lpos is not None:
                        lidar_c = max(lidar_c, lpos)
                    if rpos is not None:
                        lidar_c = min(lidar_c, rpos)
                    if abs(lidar_c - self.cfg.width // 2) > abs(center - self.cfg.width // 2):
                        center = lidar_c
                        mode = mode + "+LIDAR"

            self.ar_prev = ar_now   # ★ 매 프레임 끝에 현재 상태를 저장 (다음 프레임 비교용)

            now = time.time()

            if not self.cfg.enable_special_zone:
                # ===== 기존 차선주행 경로 (특수구역 off) =====
                angle = self.pid_controller.compute_pid_angle(center)
                self.xycar_driver.drive(angle, self.cfg.speed)
            else:
                self.special_zone_controller.update_zone(now)
                angle, speed = self.special_zone_controller.drive_special(now, center)
                self.xycar_driver.drive(angle, speed)
                
            if self.show_debug:
                # 특수구역 상태/카운트다운 (검출 오버레이는 detector 노드 자체 창)
                if self.cfg.enable_special_zone:
                    self.special_zone_controller.draw_status(frame, now)
                self.image_processor.draw_debug(frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos, corner, left_slope)

            count += 1
            if count % 30 == 0:
                half_str = "%.0f" % self.image_processor.ema_half_width if self.image_processor.ema_half_width is not None else "-"
                print("state=%s mode=%s center=%d ema_half=%s"
                    % (self.special_zone_controller.drive_state, mode, center, half_str))

            self.rate.sleep()

    def shutdown(self):
        self.xycar_driver.shutdown()
        self.image_processor.shutdown()

if __name__ == '__main__':
    xycar_controller = XycarController()
    xycar_controller.run()