import rospy
from std_msgs.msg import String
from msg_send.msg import my_msg

def callback(msg):
    reply = "Good afternoon, " + msg.last_name + " " + msg.first_name
    print(reply)
    pub.publish(reply)

rospy.init_node('remote_teacher', anonymous=True)
pub = rospy.Publisher('msg_from_xycar', String, queue_size=10)
sub = rospy.Subscriber('msg_to_xycar', my_msg, callback)
rospy.spin()
