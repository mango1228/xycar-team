#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 리눅스 PC에서 실행 - msg_to_xycar 발행 후 msg_from_xycar 출력

import rospy
from msg_send.msg import to_xycar
from msg_send.msg import from_xycar

def callback(msg):
    print(msg.data)

rospy.init_node('student')
pub = rospy.Publisher('msg_to_xycar', to_xycar, queue_size=10)
sub = rospy.Subscriber('msg_from_xycar', from_xycar, callback)

msg = to_xycar()
msg.first_name = "Sungmin"
msg.last_name = "Her"
msg.id_number = 20209876
msg.phone_number = "010-8950-1010"

rate = rospy.Rate(1)
while not rospy.is_shutdown():
    pub.publish(msg)
    rate.sleep()
