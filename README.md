# 지름길 test

## lane_drive.py
### ARtagDetector
- /ar_pose_marker 구독 -> tag 인식했을 때만 동작
- 0.5초 이내에 본 기록이 있으면 본 것으로 간주 (def ar_detected)

## main.py
- ar이 보이면 LiDAR에 의존(center = lidar_c)
- ar이 보이다가 안 보이면 일정시간 동안 LiDAR에 의존(지름길 진입 전)

## 추후 계획
- ar 인식은 잘 되지만 지름길 바깥쪽으로 가는 문제
- 지름길 내부에서 oscillation 심하여 장애물 충돌이 일어나는 문제
