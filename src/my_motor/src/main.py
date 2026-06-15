#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import rospy

from config import Config
from controller import PIDController, SpecialZoneController
from lane_drive import (
    ARtagDetector,
    ImageProcessor,
    LaneFollower,
    LidarProcessor,
    ObstacleBoundaryFollower,
    XycarDriver,
)


class XycarController:
    STATE_NORMAL = "NORMAL"
    STATE_LEFT_BOUNDARY = "LEFT_BOUNDARY"
    STATE_BOTH_BOUNDARY = "BOTH_BOUNDARY"

    def __init__(self):
        rospy.init_node("lane_drive")

        self.cfg = Config()
        self.show_debug = self.cfg.show_debug

        if self.show_debug and not os.environ.get("DISPLAY"):
            rospy.logwarn("DISPLAY 없음 - 디버그 창 비활성화")
            self.show_debug = False

        self.pid_controller = PIDController(self.cfg)
        self.image_processor = ImageProcessor(self.cfg)
        self.lidar_processor = LidarProcessor(self.cfg)
        self.xycar_driver = XycarDriver()
        self.lane_follower = LaneFollower(self.cfg)
        self.special_zone_controller = SpecialZoneController(
            self.cfg, self.pid_controller
        )
        self.ar_tag_detector = ARtagDetector(self.cfg)
        self.boundary_follower = ObstacleBoundaryFollower(self.cfg)

        self.mission_state = self.STATE_NORMAL
        self.ar_seen_once = False
        self.ar_prev_visible = False

        rospy.on_shutdown(self.shutdown)
        self.rate = rospy.Rate(30)

    def _change_mission_state(self, new_state, reason):
        if self.mission_state == new_state:
            return

        rospy.loginfo(
            "[AR boundary] state: %s -> %s (%s)" %
            (self.mission_state, new_state, reason)
        )
        self.mission_state = new_state
        self.pid_controller.reset_pid()

    def _normal_drive_center(self, camera_center, lpos, rpos, lidar_center):
        center = camera_center
        mode_suffix = ""

        if lidar_center is not None:
            corrected_lidar_center = lidar_center

            if lpos is not None:
                corrected_lidar_center = max(corrected_lidar_center, lpos)
            if rpos is not None:
                corrected_lidar_center = min(corrected_lidar_center, rpos)

            if abs(corrected_lidar_center - self.cfg.width // 2) > \
                    abs(center - self.cfg.width // 2):
                center = corrected_lidar_center
                mode_suffix = "+LIDAR"

        return center, mode_suffix

    def _compute_drive_command(self, now, center, requested_speed, boundary_safe):
        if not self.cfg.enable_special_zone:
            angle = self.pid_controller.compute_pid_angle(center)
            speed = requested_speed
        else:
            self.special_zone_controller.update_zone(now)
            angle, speed = self.special_zone_controller.drive_special(now, center)

            if self.special_zone_controller.drive_state == \
                    self.special_zone_controller.STATE_DRIVE:
                speed = requested_speed

        if not boundary_safe:
            angle = 0
            speed = 0

        return angle, speed

    def run(self):
        rospy.sleep(2.0)
        self.special_zone_controller.node_start_time = time.time()
        rospy.loginfo("lane_drive started")

        count = 0
        while not rospy.is_shutdown():
            image = self.image_processor.image
            if image.size == 0:
                self.rate.sleep()
                continue

            frame = image.copy()
            (
                camera_center,
                camera_mode,
                left,
                right,
                lpos,
                rpos,
                corner,
                left_slope,
            ) = self.image_processor.process(frame)

            scan = self.lidar_processor.lidar_scan
            roi_data = self.lidar_processor.get_roi_data(scan)
            gaps = self.lidar_processor.get_gaps(roi_data)

            lidar_center = None
            bisector = None
            if gaps:
                lidar_center, bisector = self.lane_follower.correct_lane(
                    gaps, self.lidar_processor.roi_ang_center
                )

            self.lidar_processor.publish_roi_markers(
                roi_data, gaps, bisector, scan
            )

            ar_visible = self.ar_tag_detector.ar_detected
            if ar_visible:
                self.ar_seen_once = True

            ar_just_disappeared = (
                self.ar_seen_once and
                self.ar_prev_visible and
                not ar_visible
            )

            if self.mission_state == self.STATE_NORMAL and ar_just_disappeared:
                self.boundary_follower.reset()
                self._change_mission_state(
                    self.STATE_LEFT_BOUNDARY,
                    "AR tag disappeared"
                )

            center = camera_center
            mode = camera_mode
            requested_speed = self.cfg.speed
            boundary_safe = True
            boundary_result = None

            if self.mission_state == self.STATE_NORMAL:
                center, mode_suffix = self._normal_drive_center(
                    camera_center, lpos, rpos, lidar_center
                )
                mode = camera_mode + mode_suffix

            elif self.mission_state == self.STATE_LEFT_BOUNDARY:
                boundary_result = self.boundary_follower.compute(
                    scan, ObstacleBoundaryFollower.MODE_LEFT
                )
                center = boundary_result["center"]
                mode = boundary_result["source"]
                requested_speed = self.cfg.boundary_speed
                boundary_safe = (
                    boundary_result["valid"] or boundary_result["held"]
                )

                if boundary_result["both_confirmed"]:
                    self._change_mission_state(
                        self.STATE_BOTH_BOUNDARY,
                        "left/right obstacle within %.2f m for %d frames" %
                        (
                            self.cfg.boundary_side_trigger_distance,
                            self.cfg.boundary_both_confirm_frames,
                        )
                    )

                    boundary_result = self.boundary_follower.compute(
                        scan, ObstacleBoundaryFollower.MODE_BOTH
                    )
                    center = boundary_result["center"]
                    mode = boundary_result["source"]
                    boundary_safe = (
                        boundary_result["valid"] or boundary_result["held"]
                    )

            else:
                boundary_result = self.boundary_follower.compute(
                    scan, ObstacleBoundaryFollower.MODE_BOTH
                )
                center = boundary_result["center"]
                mode = boundary_result["source"]
                requested_speed = self.cfg.boundary_speed
                boundary_safe = (
                    boundary_result["valid"] or boundary_result["held"]
                )

            now = time.time()
            angle, speed = self._compute_drive_command(
                now, center, requested_speed, boundary_safe
            )
            self.xycar_driver.drive(angle, speed)

            if self.show_debug:
                if self.cfg.enable_special_zone:
                    self.special_zone_controller.draw_status(frame, now)

                debug_lidar_center = lidar_center
                if boundary_result is not None:
                    debug_lidar_center = boundary_result["center"]

                self.image_processor.draw_debug(
                    frame,
                    center,
                    "%s/%s" % (self.mission_state, mode),
                    left,
                    right,
                    camera_center,
                    debug_lidar_center,
                    lpos,
                    rpos,
                    corner,
                    left_slope,
                )

            self.ar_prev_visible = ar_visible

            count += 1
            if count % 30 == 0:
                half_str = (
                    "%.0f" % self.image_processor.ema_half_width
                    if self.image_processor.ema_half_width is not None
                    else "-"
                )

                if boundary_result is None:
                    rospy.loginfo(
                        "mission=%s ar=%s mode=%s center=%d angle=%d speed=%d ema_half=%s" %
                        (
                            self.mission_state,
                            str(ar_visible),
                            mode,
                            center,
                            int(angle),
                            int(speed),
                            half_str,
                        )
                    )
                else:
                    rospy.loginfo(
                        "mission=%s mode=%s center=%d angle=%d speed=%d "
                        "left=%s right=%s near_count=%d safe=%s" %
                        (
                            self.mission_state,
                            mode,
                            center,
                            int(angle),
                            int(speed),
                            str(boundary_result["left_lateral"]),
                            str(boundary_result["right_lateral"]),
                            int(boundary_result["both_near_count"]),
                            str(boundary_safe),
                        )
                    )

            self.rate.sleep()

    def shutdown(self):
        self.xycar_driver.shutdown()
        self.image_processor.shutdown()


if __name__ == "__main__":
    controller = XycarController()
    controller.run()
