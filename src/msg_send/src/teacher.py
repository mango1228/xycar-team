#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

rospy.init_node('teacher')
pub = rospy.Publisher('msg_to_students', String, queue_size=10)
rate = rospy.Rate(2)  # 초당 2번

while not rospy.is_shutdown():
    pub.publish('call me please')
    rate.sleep()
