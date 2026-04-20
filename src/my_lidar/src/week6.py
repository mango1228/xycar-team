#!/usr/bin/env python

import rospy
import time
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor


class LidarDriver:
    def __init__(self):
        rospy.Subscriber('/scan', LaserScan, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)

        self.motor_msg = xycar_motor()
        self.lidar_points = None

    def callback(self, data):
        self.lidar_points = data.ranges

    def drive_go(self):
        self.motor_msg.speed = 5
        self.motor_msg.angle = 0
        self.pub.publish(self.motor_msg)

    def drive_stop(self):
        self.motor_msg.speed = 0
        self.motor_msg.angle = 0
        self.pub.publish(self.motor_msg)


class LidarSteering:
    def __init__(self):
        self.motor_control = xycar_motor()
        self.pub = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)

    def motor_pub(self, angle, speed):
        self.motor_control.angle = angle
        self.motor_control.speed = speed
        self.pub.publish(self.motor_control)

    def avoid_obstacle(self):
        speed = 3

        # 왼쪽 회피
        for i in range(40):
            self.motor_pub(-50, speed)
            rospy.sleep(0.1)

        # 직진
        for i in range(30):
            self.motor_pub(0, speed)
            rospy.sleep(0.1)

        # 오른쪽 복귀
        for i in range(40):
            self.motor_pub(50, speed)
            rospy.sleep(0.1)

        # 직진
        for i in range(30):
            self.motor_pub(0, speed)
            rospy.sleep(0.1)


class MainDriver:
    def __init__(self):
        rospy.init_node('lidar_main')

        self.lidar_driver = LidarDriver()
        self.lidar_steering = LidarSteering()

        self.rate = rospy.Rate(5)

        # 상태 변수
        self.obstacle_detected = False
        self.timer_start = None

    def run(self):
        # lidar 데이터 올 때까지 대기
        while self.lidar_driver.lidar_points is None:
            rospy.sleep(0.1)

        while not rospy.is_shutdown():
            ok = 0

            # ROI 검사 (앞쪽 ±45도)
            for degree in range(0, 46):
                if (0.01 < self.lidar_driver.lidar_points[540 + degree] <= 0.3):
                    ok += 1
                    print(f"Obstacle detected at {540 + degree} degrees...")

                if (0.01 < self.lidar_driver.lidar_points[540 - degree] <= 0.3):
                    ok += 1
                    print(f"Obstacle detected at {540 - degree} degrees...")

                # 장애물 존재
                if ok > 5:
                    self.lidar_driver.drive_stop()
                    self.timer_start = rospy.Time.now()
                    self.obstacle_detected = True
                    print("Obstacle detected: ok > 5")

                    if not self.obstacle_detected:
                        self.timer_start = rospy.Time.now()
                        self.obstacle_detected = True

                    elif self.obstacle_detected and (rospy.Time.now() - self.timer_start).to_sec() > 5.0:
                        self.lidar_steering.avoid_obstacle()
                        # 회피 후 초기화
                        self.obstacle_detected = False
                        self.timer_start = None
                else:
                    self.obstacle_detected = False
                    self.timer_start = None
                    self.lidar_driver.drive_go()

            self.rate.sleep()


if __name__ == '__main__':
    main_driver = MainDriver()
    main_driver.run()