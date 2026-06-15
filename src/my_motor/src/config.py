# -*- coding: utf-8 -*-

import rospy

class Config:
    def __init__(self):

        self.width = 640
        self.height = 480

        self.lidar_roi_x=rospy.get_param("~lidar_roi_x", 0.33)
        self.lidar_roi_y_min=rospy.get_param("~lidar_roi_y_min", -0.45)
        self.lidar_roi_y_max=rospy.get_param("~lidar_roi_y_max", -0.05)

        self.r_viz=rospy.get_param("~r_viz", 0.7)

        self.min_gap_ang=rospy.get_param("~min_gap_ang", 0.05)

        self.lidar_center_gain=rospy.get_param("~lidar_center_gain", -170.0)

        self.canny_low=rospy.get_param("~canny_low", 40)
        self.canny_high=rospy.get_param("~canny_high", 100)
        self.offset=rospy.get_param("~offset", 330)
        self.gap=rospy.get_param("~gap", 110)

        self.gain=rospy.get_param("~gain", 0.25)
        self.gain_i=rospy.get_param("~gain_i", 0.01)
        self.gain_d=rospy.get_param("~gain_d", 0.01)

        self.speed=rospy.get_param("~speed", 5.0)
        self.ema_alpha=rospy.get_param("~ema_alpha", 0.3)
        self.one_lane_ratio=rospy.get_param("~one_lane_ratio", 1.0)
        self.corner_left_base=rospy.get_param("~corner_left_base", -0.75)
        self.corner_slope_thresh=rospy.get_param("~corner_slope_thresh", 0.4)
        self.corner_shift_px=rospy.get_param("~corner_shift_px", 70)

        self.hough_threshold=rospy.get_param("~hough_threshold", 30)
        self.hough_min_len=rospy.get_param("~hough_min_len", 20)
        self.hough_max_gap=rospy.get_param("~hough_max_gap", 10)

        self.center_margin=rospy.get_param("~center_margin", 90)

        self.show_debug = rospy.get_param("~show_debug", True)

        # ===== 특수구역(횡단보도/빗금) 정지 =====
        # 마스터 토글 (false면 특수구역 감지/상태머신 미진입, 기존 차선주행과 동일)
        self.enable_special_zone = rospy.get_param("~enable_special_zone", False)
        # 특수구역 검출 노드(special_zone_detector) 실행 주기(Hz). 주행 루프와 분리됨.
        self.detect_hz = float(rospy.get_param("~detect_hz", 15.0))
        # PID 적분 와인드업 클램프 (±)
        self.i_clamp = rospy.get_param("~i_clamp", 300.0)

        # 감지 ROI (화면 하단 중앙). int() 보장: 슬라이스 인덱스/ cv2 좌표로 쓰여 float면 크래시
        # AR 태그도 이 ROI 안에 투영되는 것만 인식 (special_zone_detector와 동일 박스)
        self.detect_roi_top    = int(rospy.get_param("~detect_roi_top", 310))
        self.detect_roi_bottom = int(rospy.get_param("~detect_roi_bottom", 420))
        self.detect_roi_left   = int(rospy.get_param("~detect_roi_left", 150))
        self.detect_roi_right  = int(rospy.get_param("~detect_roi_right", 490))

        # AR 태그 3D pose(미터) -> 픽셀 투영용 카메라 내부 파라미터
        # 기본값 = usb_cam.yaml camera_matrix (fx, fy, cx, cy)
        self.cam_fx = rospy.get_param("~cam_fx", 340.876013)
        self.cam_fy = rospy.get_param("~cam_fy", 341.790625)
        self.cam_cx = rospy.get_param("~cam_cx", 333.212353)
        self.cam_cy = rospy.get_param("~cam_cy", 241.433953)
        # AR 태그 인정 최대 거리(m). 기본은 사실상 무제한(ROI 필터만 적용).
        # 가까울 때만 반응시키려면 0.5 등으로 낮춰라.
        self.ar_max_dist = rospy.get_param("~ar_max_dist", 99.0)

        # AR 태그 인식 ROI (특수구역 ROI와 별개). 기본: 화면 하단 40%, 좌우 전체.
        # height=480 기준 top=288(=480*0.6) ~ bottom=480, left=0 ~ right=640
        self.ar_roi_top    = int(rospy.get_param("~ar_roi_top", 288))
        self.ar_roi_bottom = int(rospy.get_param("~ar_roi_bottom", 480))
        self.ar_roi_left   = int(rospy.get_param("~ar_roi_left", 0))
        self.ar_roi_right  = int(rospy.get_param("~ar_roi_right", 640))
        # ===== AR 태그 인식 후 시퀀스 (순차 단계) =====
        # [0~adv] 차선추종 전진 → [adv~adv+stop] 정지 → [~+leftmost] 왼쪽 부채꼴 추종
        self.ar_advance_sec  = rospy.get_param("~ar_advance_sec", 0.5)   # 차선 따라 전진
        self.ar_stop_sec     = rospy.get_param("~ar_stop_sec", 1.0)      # 정지
        self.ar_leftmost_sec = rospy.get_param("~ar_leftmost_sec", 2.0)  # 왼쪽 부채꼴 추종+ROI확대
        # 왼쪽 부채꼴 후보 최소 각폭(도). 이 이상 벌어진 부채꼴 중 가장 왼쪽 선택
        self.ar_leftmost_min_deg = rospy.get_param("~ar_leftmost_min_deg", 15.0)
        # 왼쪽 부채꼴 추종 구간([adv+stop ~ +leftmost]) 동안 라이다 ROI 배율
        # ar_roi_scale: 좌우 폭(x) 배율 / ar_roi_forward_scale: 전방 먼 경계(y_min) 배율
        self.ar_roi_scale = rospy.get_param("~ar_roi_scale", 2.0)
        self.ar_roi_forward_scale = rospy.get_param("~ar_roi_forward_scale", 1.5)
        # AR 인식 후 이 시간(초) 동안 특별구역(횡단보도/빗금) 정지 차단
        self.ar_special_block_sec = rospy.get_param("~ar_special_block_sec", 20.0)
        # AR 인식 후 이 시간(초) 동안 카메라 추종점 미사용, 라이다 추종점만 사용
        self.ar_lidar_only_sec = rospy.get_param("~ar_lidar_only_sec", 10.0)
        # 왼쪽 부채꼴 추종 구간 동안 추종 계수(lidar_center_gain) 배율
        self.ar_leftmost_gain_scale = rospy.get_param("~ar_leftmost_gain_scale", 2.0)

        # 횡단보도 감지
        self.cross_white_thresh   = rospy.get_param("~cross_white_thresh", 180)
        self.cross_dark_thresh    = rospy.get_param("~cross_dark_thresh", 70)
        self.cross_blob_min_area  = rospy.get_param("~cross_blob_min_area", 80)
        self.cross_blob_max_area  = rospy.get_param("~cross_blob_max_area", 2500)
        self.cross_blob_max_w     = rospy.get_param("~cross_blob_max_w", 110)
        self.cross_blob_max_h     = rospy.get_param("~cross_blob_max_h", 70)
        self.cross_row_band       = rospy.get_param("~cross_row_band", 22)
        self.cross_row_span       = rospy.get_param("~cross_row_span", 120)
        self.cross_min_blocks     = rospy.get_param("~cross_min_blocks", 4)
        self.crosswalk_stop_sec   = rospy.get_param("~crosswalk_stop_sec", 10.0)
        # 정지 중 횡단보도가 이 시간(초) 이상 안 보이면 오탐 판단 → 재출발.
        # 999 = 검증(오탐 복귀) 비활성(항상 10초 정지). 켜려면 0.5 정도로.
        self.cross_lost_sec       = rospy.get_param("~cross_lost_sec", 999.0)
        self.cross_confirm_frames = rospy.get_param("~cross_confirm_frames", 1)

        # 어두운 영역 내부 흰 무늬 (횡단보도/빗금 공통)
        self.hatch_dark_thresh     = rospy.get_param("~hatch_dark_thresh", 80)
        self.hatch_white_thresh    = rospy.get_param("~hatch_white_thresh", 150)
        self.hatch_inside_min_area = rospy.get_param("~hatch_inside_min_area", 2500)

        # 판정: 흰 비율 >= classify_white_pct 면 내부 선분 평균각(0도=수평,90도=수직)으로 구분
        #       평균각 >= classify_angle_deg → 횡단보도(세로) / < → 빗금(사선)
        self.classify_white_pct  = rospy.get_param("~classify_white_pct", 7.0)
        self.classify_angle_deg  = rospy.get_param("~classify_angle_deg", 35.0)
        # 이 각도(도) 미만의 거의 수평인 선분은 평균각 계산에서 제외 (노이즈 무시)
        self.stripe_min_angle_deg = rospy.get_param("~stripe_min_angle_deg", 10.0)
        # 내부 선분 검출용 HoughLinesP 파라미터
        self.stripe_hough_threshold = rospy.get_param("~stripe_hough_threshold", 15)
        self.stripe_min_len         = rospy.get_param("~stripe_min_len", 20)
        self.stripe_max_gap         = rospy.get_param("~stripe_max_gap", 5)
        self.hatch_confirm_frames  = rospy.get_param("~hatch_confirm_frames", 2)
        # 빗금 흐름: 감지 → 차선 따라 hatch_advance_sec 동안 전진 → 영구정지
        self.hatch_advance_sec     = rospy.get_param("~hatch_advance_sec", 1.5)
        # (미사용) 옛 검증 흐름 파라미터 — 호환 위해 남겨둠
        self.hatch_verify_sec      = rospy.get_param("~hatch_verify_sec", 1.0)
        self.hatch_lost_sec        = rospy.get_param("~hatch_lost_sec", 999.0)
        # 빗금 확정 카운터가 견딜 연속 미검출 수(검출 패스 단위). 깜빡임에 정지가 리셋되는 것 방지
        self.hatch_miss_tolerance  = int(rospy.get_param("~hatch_miss_tolerance", 3))

        # 시작 직후 오탐 방지 유예 시간(초)
        self.startup_grace_sec = rospy.get_param("~startup_grace_sec", 3.0)
        # 횡단보도 10초 정지 성공 후 유예 시간(초): 그 동안 횡단보도/빗금 둘 다 감지 무시
        # (같은 횡단보도를 완전히 지나갈 때까지 충분히 길게)
        self.post_resume_grace_sec = rospy.get_param("~post_resume_grace_sec", 15.0)