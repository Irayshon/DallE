#include "my_modern_robotics/fk.h"
#include "my_modern_robotics/ik.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

TEST(IKTest, BodyAndSpaceExample) {
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

  Eigen::Matrix4d T;
  T << 0, 1, 0, -5,
       1, 0, 0, 4,
       0, 0, -1, 1.68584073,
       0, 0, 0, 1;

  Eigen::VectorXd thetalist0(3);
  thetalist0 << 1.5, 2.5, 3.0;

  Eigen::VectorXd thetalist_body = thetalist0;
  bool success_body = mymr::IK::IKinBody(Blist, M, T, thetalist_body, 1e-3, 1e-3);
  EXPECT_TRUE(success_body);
  auto T_body = mymr::FK::FKinBody(M, Blist, thetalist_body);

  Eigen::VectorXd thetalist_space = thetalist0;
  bool success_space = mymr::IK::IKinSpace(Slist, M, T, thetalist_space, 1e-3, 1e-3);
  EXPECT_TRUE(success_space);
  auto T_space = mymr::FK::FKinSpace(M, Slist, thetalist_space);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(T_body(r, c), T(r, c), 1e-3);
      EXPECT_NEAR(T_space(r, c), T(r, c), 1e-3);
    }
  }
}

TEST(IKTest, PlanarTwoLinkTarget) {
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

  double theta1 = pi / 4.0;
  double theta2 = -pi / 3.0;
  double theta12 = theta1 + theta2;
  double c12 = std::cos(theta12);
  double s12 = std::sin(theta12);
  double x = std::cos(theta1) + std::cos(theta12);
  double y = std::sin(theta1) + std::sin(theta12);

  Eigen::Matrix4d T;
  T << c12, -s12, 0, x,
       s12, c12, 0, y,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::VectorXd thetalist0(2);
  thetalist0 << 0.7, -1.0;

  Eigen::VectorXd thetalist_body = thetalist0;
  bool success_body = mymr::IK::IKinBody(Blist, M, T, thetalist_body, 1e-4, 1e-4);
  EXPECT_TRUE(success_body);
  auto T_body = mymr::FK::FKinBody(M, Blist, thetalist_body);

  Eigen::VectorXd thetalist_space = thetalist0;
  bool success_space = mymr::IK::IKinSpace(Slist, M, T, thetalist_space, 1e-4, 1e-4);
  EXPECT_TRUE(success_space);
  auto T_space = mymr::FK::FKinSpace(M, Slist, thetalist_space);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(T_body(r, c), T(r, c), 1e-4);
      EXPECT_NEAR(T_space(r, c), T(r, c), 1e-4);
    }
  }
}
