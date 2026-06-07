#!/usr/bin/env python

import math
import numpy as np
from shapely import points
from sklearn import cluster
import rospy
import time
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers


class Lidar:
    def __init__(self):
        rospy.Subscriber('/scan', LaserScan, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)

        self.motor_msg = xycar_motor()
        self.lidar_points = None
        self.angel_min = None
        self.angle_increment = None
        self.left_points = []
        self.right_points = []
        self.steering = 0

    def callback(self, data):       
        self.lidar_points = list(data.ranges)
        self.angel_min = data.angle_min
        self.angle_increment = data.angle_increment

    def roi_filter(self):
        # Set points outside the range [0.2, 1.2] to infinity
        for i in range(0,46):
            if self.lidar_points[180 + i] < 0.2 or self.lidar_points[180 + i] > 1.2:
                self.lidar_points[180 + i] = float('inf')

            if self.lidar_points[180 - i] < 0.2 or self.lidar_points[180 - i] > 1.2:
                self.lidar_points[180 - i] = float('inf')

    def process_lidar(self):
        # make clusters of consecutive points that are not infinity
        # set left_points and right_points based on the clusters
        clusters = []
        current_cluster = []

        for i, r in enumerate(self.lidar_points):
            if np.isinf(r):
                if len(current_cluster) > 5:  # Only consider clusters with more than 5 points
                    clusters.append(current_cluster)
                current_cluster = []
            else:
                current_cluster.append(i) # Store the index of the point

        if len(current_cluster) > 5:
            clusters.append(current_cluster)
            print(len(clusters)) #debug

        for cluster in clusters:
            center_idx = int(sum(cluster)/len(cluster))
            r = self.lidar_points[center_idx]
            angle = self.angel_min + center_idx * self.angle_increment

            x = -r * math.cos(angle)
            y = -r * math.sin(angle)

            if y > 0:
                self.left_points.append((x,y))
            else:
                self.right_points.append((x,y))

    def get_mid_point(self):
        if len(self.left_points) >= 2:
            left_np = np.array(self.left_points)
            x_left = left_np[:,0]
            y_left = left_np[:,1]
            a_left, b_left = np.polyfit(x_left, y_left, 1)
            #print("left :", a_left, b_left)
        else:
            return

        if len(self.right_points) >= 2:
            right_np = np.array(self.right_points)
            x_right = right_np[:,0]
            y_right = right_np[:,1]
            a_right, b_right = np.polyfit(x_right, y_right, 1)
            print("right :", a_right, b_right)
        else:
            return
        
        # One waypoint (x=0.8) on the mid line between the two fitted lines
        target_x = 0.8
        y_left = a_left*target_x + b_left
        y_right = a_right*target_x + b_right
        target_y = (y_left + y_right)/2

        steering_rad = math.atan2(target_y, target_x)
        steering_deg = math.degrees(steering_rad)
        self.steering = max(-50, min(50, steering_deg))

    def drive_go(self):
        ### test the speed
        self.motor_msg.speed = 5
        self.motor_msg.angle = self.steering
        self.pub.publish(self.motor_msg)

    def drive_stop(self):
        self.motor_msg.speed = 0
        self.motor_msg.angle = 0
        self.pub.publish(self.motor_msg)


class ARtag:
    def __init__(self):
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.callback, queue_size=1)
    
    def callback(self):   
        self.num = 0

    def get_info(self):
            self.num += 1
            print(self.num) #debug

        if self.num == 2:
            pass

# if __name__ == '__main__':
#     rospy.init_node('lidar_node')
#     lidar = Lidar()
#     artag = ARtag()
#     rate = rospy.Rate(10)

#     while not rospy.is_shutdown():
#         if lidar.lidar_points is not None:
#             lidar.roi_filter()
#             lidar.process_lidar()
#             lidar.get_mid_point()
#             lidar.drive_go()
#         else:
#             lidar.drive_stop()

#         rate.sleep()