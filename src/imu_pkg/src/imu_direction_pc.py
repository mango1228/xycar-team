#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


class ImuDirectionNode:
    def __init__(self):
        rospy.init_node('imu_direction_pc_node', anonymous=False)

        self.imu_topic = rospy.get_param('~imu_topic', '/imu')
        self.direction_topic = rospy.get_param('~direction_topic', '/direction')
        self.threshold_deg = rospy.get_param('~threshold_deg', 5.0)

        self.current_yaw = None
        self.prev_yaw = None
        self.last_imu_time = None

        self.pub = rospy.Publisher(self.direction_topic, String, queue_size=10)
        self.sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

        rospy.loginfo("imu_direction_pc_node started")
        rospy.loginfo("subscribing imu topic   : %s", self.imu_topic)
        rospy.loginfo("publishing direction    : %s", self.direction_topic)
        rospy.loginfo("turn threshold(deg/1sec): %.2f", self.threshold_deg)

    def imu_callback(self, msg):
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])

        self.current_yaw = yaw
        self.last_imu_time = rospy.Time.now()

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def timer_callback(self, _event):
        if self.current_yaw is None:
            rospy.logwarn_throttle(2.0, "No IMU data received yet.")
            return

        if self.prev_yaw is None:
            self.prev_yaw = self.current_yaw
            return

        delta_yaw = self.normalize_angle(self.current_yaw - self.prev_yaw)
        delta_deg = math.degrees(delta_yaw)

        if delta_deg > self.threshold_deg:
            direction = "left"
        elif delta_deg < -self.threshold_deg:
            direction = "right"
        else:
            direction = "straight"

        self.pub.publish(String(data=direction))

        rospy.loginfo(
            "yaw_change_1s = %.3f deg -> /direction: %s",
            delta_deg,
            direction
        )

        self.prev_yaw = self.current_yaw


if __name__ == '__main__':
    try:
        ImuDirectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass