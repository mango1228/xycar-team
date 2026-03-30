#!/usr/bin/env python
# 학생용 - my_msg 발행 후 강사의 String 회신 수신

import rospy
from std_msgs.msg import String
from msg_send.msg import my_msg

def callback(reply):
    print("From Xycar : ", reply.data)

rospy.init_node('remote_student', anonymous=True)

pub = rospy.Publisher('msg_to_xycar', my_msg, queue_size=10)
sub = rospy.Subscriber('msg_from_xycar', String, callback)

msg = my_msg()
msg.first_name = "Sungmin"
msg.last_name = "Her"
msg.id_number = 20209876
msg.phone_number = "010-8950-1010"

rate = rospy.Rate(1)
while not rospy.is_shutdown():
    pub.publish(msg)
    rate.sleep()
