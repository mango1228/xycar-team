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