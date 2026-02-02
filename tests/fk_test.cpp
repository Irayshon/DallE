#include "my_modern_robotics/fk.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

TEST(FKTest, BodyAndSpaceExample) {
  double pi = std::acos(-1.0);
  Eigen::Matrix4d M;
  M << -1, 0, 0, 0,
       0, 1, 0, 6,
       0, 0, -1, 2,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 3> Blist;
  Blist << 0, 0, 0,
           0, 0, 0,
           -1, 0, 1,
           2, 0, 0,
           0, 1, 0,
           0, 0, 0.1;

  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 0, 0, 0,
           0, 0, 0,
           1, 0, -1,
           4, 0, -6,
           0, 1, 0,
           0, 0, -0.1;

  Eigen::Vector3d thetalist;
  thetalist << pi / 2.0, 3.0, pi;

  Eigen::Matrix4d expected;
  expected << 0, 1, 0, -5,
              1, 0, 0, 4,
              0, 0, -1, 1.68584073,
              0, 0, 0, 1;

  auto T_body = mymr::FK::FKinBody(M, Blist, thetalist);
  auto T_space = mymr::FK::FKinSpace(M, Slist, thetalist);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(T_body(r, c), expected(r, c), 1e-6);
      EXPECT_NEAR(T_space(r, c), expected(r, c), 1e-6);
    }
  }
}
