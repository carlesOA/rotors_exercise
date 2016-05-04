cmake_minimum_required(VERSION 2.8.3)
project(rotors_exercise)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  cmake_modules
  dynamic_reconfigure
  tf
  mav_msgs
)

find_package(Eigen REQUIRED)

generate_dynamic_reconfigure_options(cfg/Controller.cfg)

catkin_package(
  INCLUDE_DIRS include ${Eigen_INCLUDE_DIRS}
  LIBRARIES 
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp sensor_msgs dynamic_reconfigure mav_msgs
  DEPENDS Eigen
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
  cfg/cpp
)

add_executable(estimator_node src/estimator.cpp)
add_dependencies(estimator_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(estimator_node ${catkin_LIBRARIES})


add_executable(controller_node src/control.cpp)
add_dependencies(controller_node ${catkin_EXPORTED_TARGETS})
add_dependencies(controller_node ${PROJECT_NAME}_gencfg)
target_link_libraries(controller_node ${catkin_LIBRARIES})
