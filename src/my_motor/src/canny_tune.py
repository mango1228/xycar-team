#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Canny / ROI 검증·튜닝용 디버그 노드
# - 모터 발행 안 함 → 차 안 움직임 (안전)
# - 트랙바로 Canny low/high, ROI offset 실시간 조절
# - 창 3개: 원본(ROI 박스), canny 엣지, roi 스트립(확대)

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

Width, Height = 640, 480
Gap = 40   # ROI 스트립 높이 (hough_drive.py와 동일)

image = np.empty(shape=[0])
bridge = CvBridge()

def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def nothing(x):
    pass

rospy.init_node('canny_tune')
rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)

cv2.namedWindow('canny')
cv2.createTrackbar('low',    'canny', 60,  255,        nothing)
cv2.createTrackbar('high',   'canny', 70,  255,        nothing)
cv2.createTrackbar('offset', 'canny', 340, Height-Gap, nothing)

print("canny_tune started - 트랙바로 조절, 창에서 q 누르면 종료")

frame_count = 0
rate = rospy.Rate(30)
while not rospy.is_shutdown():
    if image.size == 0:
        rate.sleep()
        continue

    frame = image.copy()
    low    = cv2.getTrackbarPos('low',    'canny')
    high   = cv2.getTrackbarPos('high',   'canny')
    offset = cv2.getTrackbarPos('offset', 'canny')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, low, high)

    roi = edge[offset:offset+Gap, 0:Width]
    roi_view = cv2.resize(roi, (Width, Gap*5), interpolation=cv2.INTER_NEAREST)

    # 원본에 ROI 위치를 초록 박스로 표시
    cv2.rectangle(frame, (0, offset), (Width-1, offset+Gap), (0, 255, 0), 2)

    cv2.imshow('original (ROI box)', frame)
    cv2.imshow('canny', edge)
    cv2.imshow('roi (enlarged)', roi_view)

    frame_count += 1
    if frame_count % 30 == 0:
        print("low=%d high=%d offset=%d | ROI 엣지픽셀=%d"
              % (low, high, offset, cv2.countNonZero(roi)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
