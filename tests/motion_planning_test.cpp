#include "my_modern_robotics/motion_planning.h"

#include <gtest/gtest.h>

#include <type_traits>

TEST(MotionPlanningTest, Placeholder) {
  static_assert(std::is_default_constructible<mymr::MotionPlanning>::value,
                "MotionPlanning should be default-constructible");
  mymr::MotionPlanning plan;
  (void)plan;
}
