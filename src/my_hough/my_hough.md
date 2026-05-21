# my_hough - 허프변환 기반 차선인식

## 개요
PDF 자료(`class/8주차 [실습] 허프변환기반 차선인식.pdf`) 기반 실습 패키지.
카메라 이미지에서 Canny Edge + Hough Transform을 사용해 좌/우 차선을 검출하고, 차선 중점과 카메라 중심의 차이를 계산한다.

## 패키지 구조
```
my_hough/
├── launch/
│   └── h_drive.launch     # xycar_motor + usb_cam + 노드 실행
├── src/
│   ├── my_hough.py        # 메인 차선인식 스크립트
│   └── line_pic*.png      # 샘플 이미지 (배치 필요)
└── my_hough.md            # 본 문서
```

**누락 (필요 시 추가)**: `package.xml`, `CMakeLists.txt` — `catkin_make` 빌드용. 단순 `python my_hough.py` 실행은 없어도 가능.

## 처리 파이프라인 (my_hough.py)
1. **이미지 로드**: `cv2.imread('line_pic1.png')`
2. **그레이스케일**: BGR → Gray
3. **블러**: `GaussianBlur (5,5)` — 노이즈 제거
4. **엣지 검출**: `Canny(30, 60)`
5. **허프 변환**: `HoughLinesP(edge, 1, π/180, 50, 50, 20)` — 직선 후보 추출
6. **기울기 필터링**: `|slope| > 0.2` 인 선분만 유지 (수평선 제거)
7. **좌/우 분리**:
   - 좌측: `slope < 0` 이고 `x2 < WIDTH/2`
   - 우측: `slope > 0` 이고 `x1 > WIDTH/2`
8. **평균 직선 추정**: 각 측 선분들의 평균 기울기/절편 계산
9. **차선 위치 산출**: `L_ROW` 라인에서 좌/우 교점 → `x_left`, `x_right`
10. **시각화**: 파란 차선, 노란 수평선, 초록 사각형(lpos/rpos/midpoint/view_center)

## 핵심 상수
```python
WIDTH, HEIGHT = 640, 480
ROI_ROW = 250                 # ROI 시작 row (원본 기준)
ROI_HEIGHT = 230              # = HEIGHT - ROI_ROW
L_ROW = 110                   # = ROI_HEIGHT - 120, 차선 위치 측정 row (ROI 내)
```

## 사진 준비
샘플 이미지를 `~/xycar-team/src/my_hough/src/` 안에 배치:
```bash
cp <보유 이미지>/line_pic1.png ~/xycar-team/src/my_hough/src/
```
- `my_hough.py` L17에서 경로 없이 파일명만 사용하므로 스크립트와 같은 폴더 필수
- 해상도 640×480 권장 (xycar 카메라 기본값)
- 여러 장 있으면 `line_pic2.png` ... 등으로 두고 코드에서 파일명 변경하며 테스트

## 코드 수정사항 (오타 수정 필요)
| Line | 현재 | 수정 |
|------|------|------|
| 22 | `cv2.GassianBlur` | `cv2.GaussianBlur` |
| 29 | `cv2.waitkey()` | `cv2.waitKey()` |
| 171 | `prev_x_rignt` | `prev_x_right` |
| 183~186 | `cv2.rectangel` | `cv2.rectangle` (4곳) |

## 실행
```bash
cd ~/xycar-team/src/my_hough/src
python my_hough.py
```
- 단계별 `imshow` 창에서 키 입력 시 다음 단계 진행
- 콘솔에 검출 직선 수, 좌/우 차선 수, 차선 중점/중심 차이 출력

## 작업 순서
1. 보유 샘플 이미지 → `src/` 폴더로 복사
2. `my_hough.py` 오타 4건 수정
3. `python my_hough.py` 실행해 결과 확인
4. (선택) `package.xml`, `CMakeLists.txt` 추가해 ROS 빌드/launch 사용

## 검증 포인트
- 각 단계 `imshow` 창 정상 표시
- 콘솔에 `Number of lines`, `Number of left/right lines` 합리적 값
- 최종 "Lanes positions" 창에 파란 차선 + 노란 수평선 + 초록 사각형 4개
- `Gap from the View_center` 값이 직관적으로 맞는지 (좌측 치우침 → 음수 등)
