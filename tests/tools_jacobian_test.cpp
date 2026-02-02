#include "my_modern_robotics/tools.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(ToolsJacobianTest, JacobianShapes) {
  Eigen::MatrixXd S(6, 2);
  S << 0, 0,
       0, 0,
       1, 1,
       0, 0,
       0, 0,
       0, 0;
  Eigen::Vector2d theta(0.1, 0.2);
  auto Js = mymr::Tools::JacobianSpace(S, theta);
  EXPECT_EQ(Js.rows(), 6);
  EXPECT_EQ(Js.cols(), 2);
}
