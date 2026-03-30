# 협업 규칙

## 브랜치 규칙
- `main` 직접 푸시 절대 금지
- `develop` 직접 푸시도 가급적 금지 (PR 권장)
- 작업 단위로 브랜치 생성: `feature/기능이름`, `fix/버그이름`

## 커밋 메시지 규칙
```
feat: 차선인식 허프변환 추가
fix: 모터 토픽 이름 오타 수정
refactor: callback 함수 정리
test: 카메라 노드 테스트 추가
docs: README 셋업 가이드 업데이트
```

## Pull Request 규칙
- PR 제목은 커밋 메시지 규칙과 동일하게
- 본인 PR은 본인이 머지 금지 (반드시 다른 팀원 1명 리뷰)
- 리뷰어는 24시간 내 리뷰 완료

## 코드 스타일
- 들여쓰기: 4 spaces
- 파일 상단에 항상 `#!/usr/bin/env python` 추가
- 노드 이름, 토픽 이름은 snake_case 사용
- 함수/변수: snake_case, 상수: UPPER_CASE

## ROS 코딩 규칙
- 노드마다 파일 1개
- launch 파일로 실행 (rosrun 직접 실행은 테스트 용도로만)
- 토픽 이름은 패키지 전체에서 통일 (아래 표 참고)

## 공통 토픽 이름
| 토픽 | 타입 | 발행자 | 구독자 |
|------|------|--------|--------|
| `/xycar_motor` | xycar_msgs/XycarMotor | control | - |
| `/usb_cam/image_raw` | sensor_msgs/Image | sensor_input | perception |
| `/imu` | sensor_msgs/Imu | sensor_input | perception |
