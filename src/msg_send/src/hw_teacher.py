#!/usr/bin/env python
# Xycar에서 실행 - msg_to_xycar 수신 후 greeting 회신

import rospy
from msg_send.msg import to_xycar
from msg_send.msg import from_xycar

def callback(msg):
    reply = from_xycar()
    reply.greeting = "Good afternoon, " + msg.last_name + " " + msg.first_name
    print(reply.greeting)
    pub.publish(reply)

rospy.init_node('teacher')
pub = rospy.Publisher('msg_from_xycar', from_xycar, queue_size=10)
sub = rospy.Subscriber('msg_to_xycar', to_xycar, callback)
rospy.spin()
