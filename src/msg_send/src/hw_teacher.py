#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Xycar에서 실행 - msg_to_xycar 수신 후 greeting 회신

import rospy
from msg_send.msg import to_xycar
from msg_send.msg import from_xycar

def callback(msg):
    reply = from_xycar()
    reply.data = "Good afternoon, " + msg.last_name + " " + msg.first_name
    print("msg.first_name = \"%s\"" % msg.first_name)
    print("msg.last_name = \"%s\"" % msg.last_name)
    print("msg.id_number = %d" % msg.id_number)
    print("msg.phone_number = \"%s\"" % msg.phone_number)
    pub.publish(reply)

rospy.init_node('teacher')
pub = rospy.Publisher('msg_from_xycar', from_xycar, queue_size=10)
sub = rospy.Subscriber('msg_to_xycar', to_xycar, callback)
rospy.spin()
