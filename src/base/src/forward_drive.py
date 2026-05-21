  #!/usr/bin/env python

  import rospy
  from xycar_msgs.msg import xycar_motor

  SPEED = 3   # 0~5, 출발하려면 최소 2~3 필요
  ANGLE = 0   # -50(좌) ~ 0(직진) ~ 50(우)
  RATE_HZ = 10  # 0.5s 무발행 시 자동정지 회피용

  rospy.init_node('forward_drive')
  pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

  msg = xycar_motor()
  rate = rospy.Rate(RATE_HZ)

  rospy.sleep(1.0)  # publisher가 master에 연결될 시간

  try:
      while not rospy.is_shutdown():
          msg.angle = ANGLE
          msg.speed = SPEED
          pub.publish(msg)
          rate.sleep()
  finally:
      msg.angle = 0
      msg.speed = 0
      pub.publish(msg)

