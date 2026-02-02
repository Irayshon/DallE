cmake_minimum_required(VERSION 3.16)
project(myModernRoboticsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build tests" ON)

find_package(Eigen3 CONFIG REQUIRED)

add_library(myModernRoboticsCpp
  src/robot_model.cpp
)

target_include_directories(myModernRoboticsCpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(myModernRoboticsCpp
  PUBLIC
    Eigen3::Eigen
)

add_executable(mr_example examples/example_main.cpp)
target_link_libraries(mr_example PRIVATE myModernRoboticsCpp)

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)

  add_executable(mr_tests tests/robot_model_test.cpp)
  target_link_libraries(mr_tests PRIVATE myModernRoboticsCpp GTest::gtest_main)
  gtest_discover_tests(mr_tests)

  add_executable(dynamics_test tests/dynamics_test.cpp)
  target_link_libraries(dynamics_test PRIVATE myModernRoboticsCpp GTest::gtest_main)
  gtest_discover_tests(dynamics_test)
endif()
