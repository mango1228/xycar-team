# Xycar 자율주행 팀 프로젝트

## 환경
- OS: Ubuntu 18.04
- ROS: Melodic Morenia
- Language: Python

## 팀 역할 분담
| 팀원 | 담당 패키지 | 역할 |
|------|------------|------|
| 1번 | `sensor_input` | 카메라, IMU 센서 입력 |
| 2번 | `perception` | 차선인식, 장애물 감지 |
| 3번 | `control` | 모터 제어, 조향 |
| 4번 | `integration` | 통합 launch, 테스트 |

## 브랜치 전략
```
main         ← 최종 완성 코드 (직접 푸시 금지)
develop      ← 통합 테스트용 브랜치
feature/xxx  ← 개인 작업 브랜치
```

### 브랜치 이름 규칙
```
feature/lane-detection
feature/motor-control
fix/camera-topic-bug
```

### 작업 흐름
1. `develop` 에서 `feature/xxx` 브랜치 생성
2. 작업 후 `develop` 으로 Pull Request
3. 팀원 1명 이상 리뷰 후 머지
4. 완성되면 `develop` → `main` 머지

## 로컬 세팅 방법

### 1. 저장소 클론
```bash
$ git clone https://github.com/[팀계정]/[저장소명].git ~/xycar_team_ws
$ cd ~/xycar_team_ws
```

### 2. ROS 환경 설정 (~/.bashrc에 추가)
```bash
source /opt/ros/melodic/setup.bash
source ~/xycar_team_ws/devel/setup.bash
alias cm='cd ~/xycar_team_ws && catkin_make'
```
```bash
$ source ~/.bashrc
```

### 3. 빌드
```bash
$ cm
```

### Xycar 원격 접속 시 추가 설정 (~/.bashrc)
```bash
export ROS_MASTER_URI=http://10.42.0.1:11311   # Xycar IP
export ROS_HOSTNAME=[내 PC IP]                  # ifconfig 로 확인
```

## 패키지 실행
```bash
$ roslaunch integration main.launch
```
