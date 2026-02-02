#include "my_modern_robotics/tools.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <type_traits>

static_assert(
    std::is_same_v<decltype(mymr::Tools::Normalize(Eigen::MatrixXd())),
                   Eigen::MatrixXd>,
    "Tools::Normalize should take and return Eigen::MatrixXd by value");

TEST(ToolsBasicTest, NearZeroAndNormalize) {
  EXPECT_TRUE(mymr::Tools::NearZero(1e-7));
  Eigen::MatrixXd v(3, 1);
  v << 1, 2, 2;
  auto n = mymr::Tools::Normalize(v);
  EXPECT_NEAR(n.norm(), 1.0, 1e-9);
}
