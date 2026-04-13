cmake_minimum_required(VERSION 3.0.2)
project(msg_send)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  my_msg.msg
  to_xycar.msg
  from_xycar.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)
