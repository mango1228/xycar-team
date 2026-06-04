#!/usr/bin/env python
# -*- coding: utf-8 -*-

# special_zone_detector.py - 특수구역(횡단보도/빗금) 검출 전용 노드
#   /usb_cam/image_raw 구독 → detect_hz 주기로 CV 검출 → /special_zone(Int8) 발행.
#   주행 노드(main.py)와 별도 프로세스로 분리되어, 무거운 CV가 주행 루프(30Hz)를 막지 않음.
#
#   /special_zone 비트마스크: bit0(1)=횡단보도 검출, bit1(2)=빗금 검출
#   (confirm 디바운스/상태머신/정지 제어는 주행 노드가 담당. 여기는 순수 검출만.)

import os
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from config import Config
from special_zone import SpecialZoneProcessor


class SpecialZoneDetector:
    def __init__(self):
        rospy.init_node('special_zone_detector')

        self.cfg = Config()
        self.show_debug = self.cfg.show_debug
        if self.show_debug and not os.environ.get('DISPLAY'):
            rospy.logwarn("DISPLAY 없음 - 검출 디버그 창 비활성화")
            self.show_debug = False

        self.bridge = CvBridge()
        self.image  = np.empty(shape=[0])
        self.sz     = SpecialZoneProcessor(self.cfg)

        rospy.Subscriber('/usb_cam/image_raw', Image, self.img_cb, queue_size=1)
        self.pub  = rospy.Publisher('/special_zone', Int8, queue_size=1)
        self.rate = rospy.Rate(self.cfg.detect_hz)

    def img_cb(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def run(self):
        print("special_zone_detector started (%.0f Hz)" % self.cfg.detect_hz)
        while not rospy.is_shutdown():
            image = self.image
            if image.size == 0:
                self.rate.sleep()
                continue

            frame = image.copy()
            cross_detected, cross_blocks, _        = self.sz.detect_crosswalk(frame)
            hatch_detected, hatch_rects, hatch_pct = self.sz.detect_hatch(frame)

            val = (1 if cross_detected else 0) | (2 if hatch_detected else 0)
            self.pub.publish(Int8(val))

            if self.show_debug:
                self.sz.draw_detect_debug(frame, cross_detected, cross_blocks,
                                          hatch_detected, hatch_pct, hatch_rects)

            self.rate.sleep()


if __name__ == '__main__':
    SpecialZoneDetector().run()
