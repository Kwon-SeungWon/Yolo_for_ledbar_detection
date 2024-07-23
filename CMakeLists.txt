cmake_minimum_required(VERSION 3.0.2)
project(yolov5_ros)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  rospy
  roscpp
  sensor_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS std_msgs rospy roscpp sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
