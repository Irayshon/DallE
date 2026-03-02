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
 

add_library(dalle_core
  src/tools.cpp
  src/trajectory.cpp
)
target_include_directories(dalle_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(dalle_core PUBLIC Eigen3::Eigen)

add_executable(quadruped_ik_commander src/quadruped_ik_commander.cpp)
ament_target_dependencies(quadruped_ik_commander rclcpp trajectory_msgs)
target_include_directories(quadruped_ik_commander PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(quadruped_ik_commander dalle_core)

if(BUILD_TESTING)
  ament_add_gtest(quadruped_targets_test test/quadruped_targets_test.cpp)
  if(TARGET quadruped_targets_test)
    target_include_directories(quadruped_targets_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()

install(TARGETS
  dalle_core
  quadruped_ik_commander
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
