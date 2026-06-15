#!/usr/bin/env python

import math
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers

# class StopLineDetector:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.bridge = CvBridge()
#         rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
#         self.detected = False
#         self.count_line = 0

#     def callback(self, data):
#         self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

#     def detect_stopline(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edge = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)

#         roi = edge[self.cfg.offset:self.cfg.offset+self.cfg.gap, 0:self.cfg.width]
#         lines = cv2.HoughLinesP(roi, 1, math.pi/180, self.cfg.hough_threshold,
#                                 minLineLength=self.cfg.hough_min_len, maxLineGap=self.cfg.hough_max_gap)

#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 if x2 == x1:
#                     continue
#                 slope = float(y2 - y1) / float(x2 - x1)
#                 if abs(slope) < 0.1:
#                     self.detected = True
#                 cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#         if self.detected:
#             self.count_line += 1

class ObstacleLaneFollower:
    def __init__(self):
        rospy.Subscriber('/scan', LaserScan, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)

        self.motor_msg = xycar_motor()
        self.lidar_points = None
        self.angle_min = None
        self.angle_increment = None
        self.steering = 0
        self.min_cluster_size = 5

    def callback(self, data):       
        self.lidar_points = list(data.ranges)
        self.angle_min = data.angle_min
        self.angle_increment = data.angle_increment

    def roi_filter(self):
        # 전방 ±45°(index 135~225)만 남기고, 그 안에서도 거리[0.2, 1.2] 밖은 inf 처리.
        # ※ index 180이 전방인지 반드시 확인할 것:
        #    angle_min ≈ -3.14 이면 index 180 = 전방 (OK)
        #    angle_min ≈ 0     이면 전방은 index 0/360 → 아래 ROI 인덱스를 바꿔야 함.
        n = len(self.lidar_points)
        print("Lidar points:", n)
        for i in range(n):
            r = self.lidar_points[i]
            if i < 135 or i > 225 or r < 0.2 or r > 1.2:
                self.lidar_points[i] = float('inf')

    def process_lidar(self):
        # make clusters of consecutive points that are not infinity
        # set left_points and right_points based on the clusters
        clusters = []
        current_cluster = []
        self.left_points = []
        self.right_points = []

        for i, r in enumerate(self.lidar_points):
            if np.isinf(r):
                if len(current_cluster) >= self.min_cluster_size:  # Only consider clusters with more than 5 points
                    clusters.append(current_cluster)
                current_cluster = []
            else:
                current_cluster.append(i) # Store the index of the point

        if len(current_cluster) >= self.min_cluster_size:  # Check the last cluster
            clusters.append(current_cluster)

        #print(len(clusters)) #debug

        for cluster in clusters:
            center_idx = int(sum(cluster)/len(cluster))
            r = self.lidar_points[center_idx]
            angle = self.angle_min + center_idx * self.angle_increment

            x = -r * math.cos(angle)
            y = -r * math.sin(angle)

            if y > 0:
                self.left_points.append((x,y))
            else:
                self.right_points.append((x,y))
        
    def get_mid_point(self):
        # One waypoint (x=0.8) on the mid line between the two fitted lines
        target_x = 0.8
        y_left = None
        y_right = None
    
        if len(self.left_points) >= 2:
            left_np = np.array(self.left_points)
            a_left, b_left = np.polyfit(left_np[:, 0], left_np[:, 1], 1)
            y_left = a_left * target_x + b_left
            #print("left :", a_left, b_left)
    
        if len(self.right_points) >= 2:
            right_np = np.array(self.right_points)
            a_right, b_right = np.polyfit(right_np[:, 0], right_np[:, 1], 1)
            y_right = a_right * target_x + b_right
            #print("right :", a_right, b_right)
        
        if y_left is not None and y_right is not None:
            target_y = (y_left + y_right) / 2.0
        elif y_left is not None:
            # 왼쪽만 보임 -> 왼쪽 경계에서 일정 간격 우측으로 (값은 통로 폭에 맞게 튜닝)
            target_y = y_left - 0.25
        elif y_right is not None:
            # 오른쪽만 보임
            target_y = y_right + 0.25
        else:
            # 양쪽 다 안 보임 -> 직진 유지
            self.steering = 0
            return self.steering

        steering_rad = math.atan2(target_y, target_x)
        steering_deg = math.degrees(steering_rad)
        self.steering = max(-50, min(50, steering_deg))

        return self.steering

    def drive_go(self):
        ### test the speed
        self.motor_msg.speed = 5
        self.motor_msg.angle = self.steering
        self.pub.publish(self.motor_msg)

    def drive_stop(self):
        self.motor_msg.speed = 0
        self.motor_msg.angle = 0
        self.pub.publish(self.motor_msg)

class ARtagDetector:
    def __init__(self):
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.callback, queue_size=1)
        self.detected = False     # 이번 프레임에 마커가 보였나
        self.marker_id = None
        self.distance = None      # 카메라 전방 거리(z)

    def callback(self, msg):      # ← 메시지 인자 필수
        if len(msg.markers) == 0:
            self.detected = False
            return
        
        self.detected = True
        
        

if __name__ == '__main__':
    rospy.init_node('lidar_node')
    obstacle_lane_follower = ObstacleLaneFollower()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if obstacle_lane_follower.lidar_points is not None:
            obstacle_lane_follower.roi_filter()
            obstacle_lane_follower.process_lidar()
            obstacle_lane_follower.get_mid_point()
            obstacle_lane_follower.drive_go()
        else:
            obstacle_lane_follower.drive_stop()

        rate.sleep()