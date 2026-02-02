#include "my_modern_robotics/tools.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(ToolsBasicTest, NearZeroAndNormalize) {
  EXPECT_TRUE(mymr::Tools::NearZero(1e-7));
  Eigen::Vector3d v(1, 2, 2);
  auto n = mymr::Tools::Normalize(v);
  EXPECT_NEAR(n.norm(), 1.0, 1e-9);
}
