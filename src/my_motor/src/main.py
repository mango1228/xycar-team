#!/usr/bin/env python

import rospy
import os
import time
from config import Config
from lane_drive import ImageProcessor, LidarProcessor, XycarDriver

class XycarController:
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

        rospy.on_shutdown(self.shutdown)  # 안전 정지 (콜백 등록: 괄호 없이 함수 참조)

        self.rate = rospy.Rate(30)

    def run(self):
        rospy.sleep(2.0)  # 센서 워밍업 대기
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

            # PID 제어
            now = time.time()
            dt  = (now - self.pid_time) if self.pid_time is not None else 1e-6
            dt  = max(dt, 1e-6)
            self.pid_time = now

            error   = center - self.cfg.width // 2
            self.i_error += error * dt
            d_out   = (error - self.prev_error) / dt
            self.prev_error = error

            angle = error * self.cfg.gain + self.i_error * self.cfg.gain_i + d_out * self.cfg.gain_d
            angle = max(-50, min(50, angle))

            self.xycar_driver.drive(angle, self.cfg.speed)

            if self.show_debug:
                self.image_processor.draw_debug(frame, center, mode, left, right, cam_center, lidar_c, lpos, rpos, corner, left_slope)

            count += 1
            if count % 30 == 0:
                half_str = "%.0f" % self.image_processor.ema_half_width if self.image_processor.ema_half_width is not None else "-"
                print("mode=%s center=%d angle=%d ema_half=%s"
                    % (mode, center, int(angle), half_str))

            self.rate.sleep()
    
    def shutdown(self):
        self.xycar_driver.shutdown()
        self.image_processor.shutdown()

if __name__ == '__main__':
    xycar_controller = XycarController()
    xycar_controller.run()
