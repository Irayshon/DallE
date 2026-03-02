cmake_minimum_required(VERSION 3.8)
project(robot_commander_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

add_library(walle_kinematics
  src/tools.cpp
  src/fk.cpp
  src/ik.cpp
)
target_include_directories(walle_kinematics PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(walle_kinematics PUBLIC Eigen3::Eigen OsqpEigen::OsqpEigen)

add_executable(xarm7_ik_commander src/xarm7_ik_commander.cpp)
ament_target_dependencies(xarm7_ik_commander rclcpp trajectory_msgs)
target_link_libraries(xarm7_ik_commander walle_kinematics)

if(BUILD_TESTING)
  ament_add_gtest(ik_test test/ik_test.cpp)
  if(TARGET ik_test)
    target_link_libraries(ik_test walle_kinematics)
  endif()
endif()

install(TARGETS
  walle_kinematics
  xarm7_ik_commander
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
