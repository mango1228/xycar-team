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

        if self.show_debug and not os.environ.get('DISPLAY'):
            rospy.logwarn("DISPLAY 없음 - 디버그 창 비활성화")
            self.show_debug = False

        self.pid_controller = PIDController(self.cfg)
        self.image_processor = ImageProcessor(self.cfg)
        self.lidar_processor = LidarProcessor(self.cfg)
        self.xycar_driver = XycarDriver()
        self.lane_follower = LaneFollower(self.cfg)
        self.special_zone_controller = SpecialZoneController(self.cfg, self.pid_controller)
        self.ar_tag_detector = ARtagDetector(self.cfg)
        # AR 태그 인식 후 시퀀스 상태
        self.ar_t0 = None          # 시퀀스 시작 시각 (None=비활성)
        self.ar_armed = True       # True일 때만 새 시퀀스 트리거 (태그 사라질 때까지 재트리거 방지)
        self.ar_roi_cur_fscale = 1.0   # 현재 적용된 ROI 전방 배율 (1.0=기본)
        # 기본 라이다 ROI 백업 (시퀀스 종료 후 복귀용)
        self.base_lidar_roi_x     = self.cfg.lidar_roi_x
        self.base_lidar_roi_y_min = self.cfg.lidar_roi_y_min
        self.base_lidar_roi_y_max = self.cfg.lidar_roi_y_max


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

            lidar_follow = lidar_c   # 카메라 클램프 전 순수 라이다 추종점 (AR 라이다전용 구간용)

            self.lidar_processor.publish_roi_markers(roi_data, gaps, bisector, self.lidar_processor.lidar_scan)
            
            # 라이다 보정: 차선 경계 안으로 클램프 (차선 바깥 조향 방지)
            if lidar_c is not None:
                if lpos is not None:
                    lidar_c = max(lidar_c, lpos)
                if rpos is not None:
                    lidar_c = min(lidar_c, rpos)
                if abs(lidar_c - self.cfg.width // 2) > abs(center - self.cfg.width // 2):
                    center = lidar_c
                    mode = mode + "+LIDAR"

            now = time.time()

            # ===== AR 태그 인식 시퀀스 트리거 (1회) =====
            ar_in = (self.ar_tag_detector.detected and
                     self.ar_tag_detector.distance < self.cfg.ar_max_dist)
            if self.ar_armed and ar_in:
                self.ar_t0 = now
                self.ar_armed = False
                print("AR detected (id=%s) -> sequence start"
                      % self.ar_tag_detector.marker_id)

            el = (now - self.ar_t0) if self.ar_t0 is not None else None

            # ===== 단계별 주행 =====
            t_adv   = self.cfg.ar_advance_sec
            t_stop  = t_adv + self.cfg.ar_stop_sec
            t_left  = t_stop + self.cfg.ar_leftmost_sec
            t_stop2 = t_left + self.cfg.ar_stop2_sec
            t_right = t_stop2 + self.cfg.ar_rightmost_sec
            t_stop3 = t_right + self.cfg.ar_stop3_sec
            t_center = t_stop3 + self.cfg.ar_center_sec

            # ROI 전방 확대: 왼쪽 추종은 ×2(_left), 오른쪽/중앙 추종은 ×1.5, 그 외엔 ×1
            if el is not None and t_stop <= el < t_left:
                desired_f = self.cfg.ar_roi_forward_scale_left
            elif el is not None and (t_stop2 <= el < t_right or t_stop3 <= el < t_center):
                desired_f = self.cfg.ar_roi_forward_scale
            else:
                desired_f = 1.0
            if desired_f != self.ar_roi_cur_fscale:
                self.lidar_processor.set_roi(self.base_lidar_roi_x * self.cfg.ar_roi_scale,
                                             self.base_lidar_roi_y_min * desired_f,
                                             self.base_lidar_roi_y_max)
                self.ar_roi_cur_fscale = desired_f

            # 특별구역(횡단보도/빗금) 정지 차단 구간: AR 인식 후 ar_special_block_sec 동안
            block_special = (el is not None and el < self.cfg.ar_special_block_sec)
            # 라이다 추종점만 사용(카메라 추종점 미사용) 구간: AR 인식 후 ar_lidar_only_sec 동안
            lidar_only = (el is not None and el < self.cfg.ar_lidar_only_sec)

            # 차단 시간까지 모두 끝나고 태그도 사라지면 재무장
            if self.ar_t0 is not None and el >= self.cfg.ar_special_block_sec and not ar_in:
                self.ar_t0 = None
                self.ar_armed = True

            if el is not None and el < t_adv:
                # [0~adv] 전진. 라이다전용 구간이면 라이다 추종점, 아니면 차선
                mode = "AR_ADVANCE"
                drive_c = lidar_follow if (lidar_only and lidar_follow is not None) else center
                angle = self.pid_controller.compute_pid_angle(drive_c)
                self.xycar_driver.drive(angle, self.cfg.speed)
            elif el is not None and el < t_stop:
                # [adv~adv+stop] 정지
                mode = "AR_STOP"
                self.xycar_driver.drive(0, 0)
            elif el is not None and el < t_left:
                # [~+leftmost] 15도 이상 부채꼴 중 가장 왼쪽 추종 (추종 계수 ×ar_leftmost_gain_scale)
                mode = "AR_LEFTMOST"
                if gaps:
                    steer_c, _ = self.lane_follower.correct_lane_leftmost(
                        gaps, self.lidar_processor.roi_ang_center,
                        self.cfg.ar_leftmost_min_deg, self.cfg.ar_leftmost_gain_scale)
                else:
                    steer_c = center
                angle = self.pid_controller.compute_pid_angle(steer_c)
                self.xycar_driver.drive(angle, self.cfg.speed)
            elif el is not None and el < t_stop2:
                # [t_left~t_stop2] 2차 정지
                mode = "AR_STOP2"
                self.xycar_driver.drive(0, 0)
            elif el is not None and el < t_right:
                # [t_stop2~t_right] 재출발: 15도 이상 부채꼴 중 가장 오른쪽 추종
                mode = "AR_RIGHTMOST"
                if gaps:
                    # 오른쪽부터는 추종 게인 일반(×1) (왼쪽만 ×2)
                    steer_c, _ = self.lane_follower.correct_lane_rightmost(
                        gaps, self.lidar_processor.roi_ang_center, self.cfg.ar_leftmost_min_deg)
                else:
                    steer_c = center
                angle = self.pid_controller.compute_pid_angle(steer_c)
                self.xycar_driver.drive(angle, self.cfg.speed)
            elif el is not None and el < t_stop3:
                # [t_right~t_stop3] 3차 정지
                mode = "AR_STOP3"
                self.xycar_driver.drive(0, 0)
            elif el is not None and el < t_center:
                # [t_stop3~t_center] 중앙 최근접 15도 이상 부채꼴 추종 (10초)
                mode = "AR_CENTER"
                if gaps:
                    # 중앙 추종은 일반 게인(×1) 사용 (왼쪽/오른쪽만 ×2)
                    steer_c, _ = self.lane_follower.correct_lane_centermost(
                        gaps, self.lidar_processor.roi_ang_center, self.cfg.ar_leftmost_min_deg)
                else:
                    steer_c = center
                angle = self.pid_controller.compute_pid_angle(steer_c)
                self.xycar_driver.drive(angle, self.cfg.speed)
            elif lidar_only:
                # [t_left~ar_lidar_only_sec] 카메라 추종점 미사용, 라이다 추종점만
                mode = "AR_LIDAR"
                drive_c = lidar_follow if lidar_follow is not None else center
                angle = self.pid_controller.compute_pid_angle(drive_c)
                self.xycar_driver.drive(angle, self.cfg.speed)
            elif self.cfg.enable_special_zone and not block_special:
                # ===== 특수구역 상태머신 (AR 차단 구간이 아닐 때만) =====
                self.special_zone_controller.update_zone(now)
                angle, speed = self.special_zone_controller.drive_special(now, center)
                self.xycar_driver.drive(angle, speed)
            else:
                # ===== 일반 차선주행 (특수구역 off 또는 AR 후 차단 구간) =====
                angle = self.pid_controller.compute_pid_angle(center)
                self.xycar_driver.drive(angle, self.cfg.speed)
                
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
