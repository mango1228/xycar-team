#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import time
import math
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
        self.ar_roi_cur_ymax = self.cfg.lidar_roi_y_max  # 현재 적용된 ROI 가까운 경계(y_max)
        self.ar_roi_cur_x = self.cfg.lidar_roi_x * self.cfg.ar_roi_scale  # 현재 적용된 ROI 좌우폭(x)
        self.ar_center_ema = None  # 중앙추종 라이다 추종점 EMA 상태 (시퀀스 시작 시 초기화)
        self.prev_mode = None      # 직전 프레임 주행 모드 (모드 전환 감지 → PID 리셋용)
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
                self.ar_center_ema = None  # 새 시퀀스: 중앙추종 스무딩 상태 초기화
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
            t_stop4 = t_center + self.cfg.ar_stop4_sec
            t_after = t_center + self.cfg.ar_after_center_sec   # 라이다축소 끝 (t_center 기준)
            t_stop5 = t_after + self.cfg.ar_stop5_sec           # 라이다축소 후 5차 정지

            # ROI 전방 배율: 왼쪽 ×2(_left), 오른쪽 ×1.5(기본), 중앙 ×0.6(_center 축소), 그 외 ×1
            if el is not None and t_stop <= el < t_left:
                desired_f = self.cfg.ar_roi_forward_scale_left
            elif el is not None and t_stop2 <= el < t_right:
                desired_f = self.cfg.ar_roi_forward_scale
            elif el is not None and t_stop3 <= el < t_center:
                desired_f = self.cfg.ar_roi_forward_scale_center
            elif el is not None and t_center <= el < t_center + self.cfg.ar_after_center_sec:
                # 중앙추종 끝나고 ar_after_center_sec 동안 전방 축소 (y_min -0.45 -> -0.25)
                desired_f = self.cfg.ar_roi_forward_scale_after
            else:
                desired_f = 1.0
            # 중앙추종 끝(t_center)까지만 y_max를 ar_roi_y_max(-0.15)로. 라이다 축소 구간부터는 기본(-0.05)
            ymax_active = (el is not None and el < t_center)
            desired_ymax = self.cfg.ar_roi_y_max if ymax_active else self.base_lidar_roi_y_max
            # 라이다축소(after) 구간만 좌우폭 ar_roi_x_after(0.15). 그 외엔 기본 0.33
            in_after = (el is not None and t_center <= el < t_center + self.cfg.ar_after_center_sec)
            desired_x = self.cfg.ar_roi_x_after if in_after else (self.base_lidar_roi_x * self.cfg.ar_roi_scale)
            if (desired_f != self.ar_roi_cur_fscale or desired_ymax != self.ar_roi_cur_ymax
                    or desired_x != self.ar_roi_cur_x):
                self.lidar_processor.set_roi(desired_x,
                                             self.base_lidar_roi_y_min * desired_f,
                                             desired_ymax)
                self.ar_roi_cur_fscale = desired_f
                self.ar_roi_cur_ymax = desired_ymax
                self.ar_roi_cur_x = desired_x

            # 특별구역(횡단보도/빗금) 정지 차단 구간: AR 인식 후 ar_special_block_sec 동안
            block_special = (el is not None and el < self.cfg.ar_special_block_sec)
            # 라이다 추종점만 사용(카메라 추종점 미사용) 구간: AR 인식 후 ar_lidar_only_sec 동안
            lidar_only = (el is not None and el < self.cfg.ar_lidar_only_sec)

            # 차단 시간까지 모두 끝나고 태그도 사라지면 재무장
            if self.ar_t0 is not None and el >= self.cfg.ar_special_block_sec and not ar_in:
                self.ar_t0 = None
                self.ar_armed = True

            angle = 0  # 이번 프레임 조향값 (정지 구간은 0 유지)
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
                # [t_stop2~t_right] 15도 이상 부채꼴 중 가장 오른쪽 추종 (게인 ×1)
                mode = "AR_RIGHTMOST"
                if gaps:
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
                bis = None
                if gaps:
                    # 중앙 추종은 게인 ×ar_center_gain_scale(기본 0.6)
                    steer_c, bis = self.lane_follower.correct_lane_centermost(
                        gaps, self.lidar_processor.roi_ang_center,
                        self.cfg.ar_leftmost_min_deg, self.cfg.ar_center_gain_scale)
                else:
                    steer_c = center
                # 라이다 추종점 EMA 스무딩: 옆 블록이 들고나며 목표가 프레임마다
                # 튀는 것을 완화 (카메라 차선 중앙과 동일한 방식). 게인 증폭 전 안정화.
                a = self.cfg.lidar_ema_alpha
                if self.ar_center_ema is None:
                    self.ar_center_ema = float(steer_c)
                else:
                    self.ar_center_ema = a * float(steer_c) + (1.0 - a) * self.ar_center_ema
                steer_c = int(self.ar_center_ema)
                angle = self.pid_controller.compute_pid_angle(steer_c)
                self.xycar_driver.drive(angle, self.cfg.speed)
                # --- 진단 로그: 부채꼴 후보(폭 w / 중앙편차 d, 단위 deg)와
                #     선택된 추종점의 dev, 최종 조향 angle. 충돌 직전 값을 보면
                #     (a) angle이 ±50 포화인지  (b) dev가 작은지(추종 대상 문제)
                #     (c) 비슷한 폭의 옆 부채꼴을 골랐는지 한눈에 확인 가능.
                c0 = self.lidar_processor.roi_ang_center
                glist = " ".join(
                    "[w%.0f d%.0f]" % (math.degrees(g[1] - g[0]),
                                       math.degrees(c0 - (g[0] + g[1]) / 2.0))
                    for g in gaps)
                chosen_dev = math.degrees(c0 - bis) if bis is not None else 0.0
                rospy.loginfo_throttle(0.2,
                    "AR_CENTER gaps=%d %s | chosen_dev=%.0f steer_c=%d angle=%d" %
                    (len(gaps), glist, chosen_dev, steer_c, angle))
            elif el is not None and el < t_stop4:
                # [t_center~t_stop4] 중앙 추종 후 4차 정지
                mode = "AR_STOP4"
                self.xycar_driver.drive(0, 0)
            elif el is not None and el < t_after:
                # [t_stop4~t_after] 지름길 탈출 구간: 카메라 차선 중앙으로 주행.
                # 빠져나오면 양쪽 차선이 정상적으로 보이므로, 프레임마다 튀는 raw
                # 라이다 추종점(가장 큰 부채꼴) 대신 안정적인 카메라 중앙을 따른다.
                # → 중앙추종(라이다)→카메라로 추종 대상이 한 번에 바뀌며 생기던
                #   급스윙/차선이탈(out)을 방지.
                mode = "AR_AFTER_CENTER"
                # 중앙추종(×1.95, EMA)에서 빠져나오는 순간, 키워놨던 PID의
                # 적분(I)·미분(D) 누적이 그대로 넘어와 전환 스파이크 → 진동을
                # 유발한다. 이 모드 진입 첫 프레임에 PID를 리셋해 와인드업 제거.
                if self.prev_mode == "AR_CENTER":
                    self.pid_controller.reset_pid()
                angle = self.pid_controller.compute_pid_angle(center)
                self.xycar_driver.drive(angle, self.cfg.speed)
            elif el is not None and el < t_stop5:
                # [t_after~t_stop5] 라이다축소 후 5차 정지
                mode = "AR_STOP5"
                self.xycar_driver.drive(0, 0)
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

            self.prev_mode = mode   # 다음 프레임의 모드 전환 감지용
            count += 1
            if count % 30 == 0:
                half_str = "%.0f" % self.image_processor.ema_half_width if self.image_processor.ema_half_width is not None else "-"
                print("state=%s mode=%s angle=%d center=%d ema_half=%s"
                    % (self.special_zone_controller.drive_state, mode, angle, center, half_str))

            self.rate.sleep()

    def shutdown(self):
        self.xycar_driver.shutdown()
        self.image_processor.shutdown()

if __name__ == '__main__':
    xycar_controller = XycarController()
    xycar_controller.run()
