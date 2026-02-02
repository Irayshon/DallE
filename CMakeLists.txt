cmake_minimum_required(VERSION 3.16)

project(myModernRoboticsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(my_modern_robotics
  src/tools.cpp
  src/fk.cpp
  src/ik.cpp
  src/dynamics.cpp
  src/inverse_dynamics.cpp
  src/trajectory.cpp
  src/robot_control.cpp
  src/motion_planning.cpp
)

target_include_directories(my_modern_robotics PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(GTest CONFIG REQUIRED)

enable_testing()

add_executable(api_smoke_test tests/api_smoke_test.cpp)
target_link_libraries(api_smoke_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(api_smoke_test)
