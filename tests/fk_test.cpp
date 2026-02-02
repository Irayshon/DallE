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

TEST(FKTest, PlanarTwoLinkExample) {
  double pi = std::acos(-1.0);
  Eigen::Matrix4d M;
  M << 1, 0, 0, 2,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 2> Slist;
  Slist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           0, -1,
           0, 0;

  Eigen::Matrix<double, 6, 2> Blist;
  Blist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           2, 1,
           0, 0;

  Eigen::Vector2d thetalist;
  thetalist << pi / 4.0, -pi / 3.0;

  double theta12 = thetalist(0) + thetalist(1);
  double c12 = std::cos(theta12);
  double s12 = std::sin(theta12);
  double x = std::cos(thetalist(0)) + std::cos(theta12);
  double y = std::sin(thetalist(0)) + std::sin(theta12);

  Eigen::Matrix4d expected;
  expected << c12, -s12, 0, x,
              s12, c12, 0, y,
              0, 0, 1, 0,
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
