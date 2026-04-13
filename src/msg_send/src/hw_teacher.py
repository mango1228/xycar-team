#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Xycar에서 실행 - msg_to_xycar 수신 후 greeting 회신

import rospy
from msg_send.msg import to_xycar
from msg_send.msg import from_xycar

def callback(msg):
    reply = from_xycar()
    reply.data = "Good afternoon, " + msg.last_name + " " + msg.first_name
<<<<<<< HEAD:src/msg_send/src/hw_teacher.py

=======
    
>>>>>>> 5d44efb17bf5f8f5d7f9c88eb07559ca28b29ed6:msg_send/src/hw_teacher.py
    print("1. Name : ", msg.last_name + msg.first_name)
    print("2. ID : ", msg.id_number)
    print("3. Phone Number : ", msg.phone_number)
    
    pub.publish(reply)

rospy.init_node('teacher')
sub = rospy.Subscriber('msg_to_xycar', to_xycar, callback)
pub = rospy.Publisher('msg_from_xycar', from_xycar, queue_size=10)
rospy.spin()
