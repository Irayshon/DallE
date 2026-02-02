cmake_minimum_required(VERSION 3.16)

project(myModernRoboticsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

find_package(Eigen3 CONFIG REQUIRED)

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

target_link_libraries(my_modern_robotics PUBLIC Eigen3::Eigen)

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)

  add_executable(api_smoke_test tests/api_smoke_test.cpp)
  target_link_libraries(api_smoke_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  add_executable(tools_basic_test tests/tools_basic_test.cpp)
  target_link_libraries(tools_basic_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  add_executable(tools_transform_test tests/tools_transform_test.cpp)
  target_link_libraries(tools_transform_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  add_executable(tools_jacobian_test tests/tools_jacobian_test.cpp)
  target_link_libraries(tools_jacobian_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  add_executable(fk_test tests/fk_test.cpp)
  target_link_libraries(fk_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  add_executable(ik_test tests/ik_test.cpp)
  target_link_libraries(ik_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  add_executable(inverse_dynamics_test tests/inverse_dynamics_test.cpp)
  target_link_libraries(inverse_dynamics_test PRIVATE my_modern_robotics GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(api_smoke_test)
  gtest_discover_tests(tools_basic_test)
  gtest_discover_tests(tools_transform_test)
  gtest_discover_tests(tools_jacobian_test)
  gtest_discover_tests(fk_test)
  gtest_discover_tests(ik_test)
  gtest_discover_tests(inverse_dynamics_test)
endif()
