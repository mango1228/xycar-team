#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

rospy.init_node('teacher')
pub = rospy.Publisher('msg_to_students', Int32, queue_size=10)
rate = rospy.Rate(2)
count = 1

while pub.get_num_connections() == 0:
    count = 1

while not rospy.is_shutdown():
    pub.publish(count)
    count = count + 1
    rate.sleep()
