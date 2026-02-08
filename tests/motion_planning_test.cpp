#include "WallE/motion_planning.h"

#include <gtest/gtest.h>

#include <type_traits>

TEST(MotionPlanningTest, Placeholder) {
  static_assert(std::is_default_constructible<WallE::MotionPlanning>::value,
                "MotionPlanning should be default-constructible");
  WallE::MotionPlanning plan;
  (void)plan;
}
