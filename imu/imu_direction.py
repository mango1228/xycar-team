#!/usr/bin/env python
import rospy
import time

from sensor_msgs.msg import Imu
from sensor_msgs.msg import direction
from tf.transformations import euler_from_quaternion

Imu_msg = None

def imu_callback(data):
  global Imu_msg
  Imu_msg = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
  (roll, pitch, yaw) = euler_from_quaternion(Imu_msg)

  data_direction = direction()

  if yaw > 0.5:
    data_direction.direction = "right"
  elif yaw < -0.5:
    data_direction.direction = "left"
  else:
    data_direction.direction = "forward"

  pub.publish(data_direction)

rospy.init_node("Imu_Print")
rospy.Subscriber("imu", Imu, imu_callback)
pub = rospy.Publisher("direction", direction, queue_size=10)

while not rospy.is_shutdown():
  if Imu_msg == None:
    continue

  

  time.sleep(1.0)
