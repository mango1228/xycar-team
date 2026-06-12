#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers


class ARTagTest:
    def __init__(self):
        rospy.init_node("ar_tag_test")

        self.detected = False

        rospy.Subscriber(
            "/ar_pose_marker",
            AlvarMarkers,
            self.callback,
            queue_size=1
        )

        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

        rospy.loginfo("AR Tag Test Started")

    def callback(self, msg):
        if len(msg.markers) > 0:
            self.detected = True
            rospy.loginfo("callback: DETECTED %d markers", len(msg.markers))

            for marker in msg.markers:
                rospy.loginfo("ID=%d", marker.id)
        else:
            self.detected = False

    def timer_callback(self, event):
        if self.detected:
            rospy.loginfo("STATUS: AR TAG DETECTED")
        else:
            rospy.loginfo("STATUS: NO AR TAG")


if __name__ == "__main__":
    ARTagTest()
    rospy.spin()




