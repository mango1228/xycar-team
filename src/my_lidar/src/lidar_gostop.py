#!/usr/bin/env python

import rospy, time
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor

motor_msg = xycar_motor()
lidar_points = None


def callback(data):
    global lidar_points, motor_msg
    lidar_points = data.ranges


def drive_go():
    global motor_msg
    motor_msg.speed = 5
    motor_msg.angle = 0
    pub.publish(motor_msg)


def drive_stop():
    global motor_msg
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)


rospy.init_node('lidar_driver')
rospy.Subscriber('/scan', LaserScan, callback, queue_size=1)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)


while lidar_points is None:
    continue

rate = rospy.Rate(5)

while not rospy.is_shutdown():
    ok = 0

    for degree in range(0, 46):
        if 0.01 < lidar_points[degree] <= 0.3:
            ok += 1

        if 0.01 < lidar_points[720 - degree] <= 0.3:
            ok += 1

    if ok > 5:
        drive_stop()
        break

    if ok <= 5:
        drive_go()

    rate.sleep()