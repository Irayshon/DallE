#include "my_modern_robotics/inverse_dynamics.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

TEST(InverseDynamicsTest, ThreeLinkExample) {
  Eigen::VectorXd thetalist(3);
  thetalist << 0.1, 0.1, 0.1;

  Eigen::VectorXd dthetalist(3);
  dthetalist << 0.1, 0.2, 0.3;

  Eigen::VectorXd ddthetalist(3);
  ddthetalist << 2.0, 1.5, 1.0;

  Eigen::Vector3d g(0.0, 0.0, -9.8);

  Eigen::VectorXd Ftip(6);
  Ftip << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  Eigen::Matrix4d M01;
  M01 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0.089159,
      0, 0, 0, 1;

  Eigen::Matrix4d M12;
  M12 << 0, 0, 1, 0.28,
      0, 1, 0, 0.13585,
      -1, 0, 0, 0,
      0, 0, 0, 1;

  Eigen::Matrix4d M23;
  M23 << 1, 0, 0, 0,
      0, 1, 0, -0.1197,
      0, 0, 1, 0.395,
      0, 0, 0, 1;

  Eigen::Matrix4d M34;
  M34 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0.14225,
      0, 0, 0, 1;

  std::vector<Eigen::MatrixXd> Mlist{M01, M12, M23, M34};

  Eigen::Matrix<double, 6, 6> G1 = Eigen::Matrix<double, 6, 6>::Zero();
  G1.diagonal() << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;

  Eigen::Matrix<double, 6, 6> G2 = Eigen::Matrix<double, 6, 6>::Zero();
  G2.diagonal() << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;

  Eigen::Matrix<double, 6, 6> G3 = Eigen::Matrix<double, 6, 6>::Zero();
  G3.diagonal() << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

  std::vector<Eigen::MatrixXd> Glist{G1, G2, G3};

  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 1, 0, 0,
      0, 1, 1,
      1, 0, 0,
      0, -0.089, -0.089,
      1, 0, 0,
      0, 0, 0.425;

  Eigen::VectorXd tau =
      mymr::InverseDynamics::Compute(thetalist, dthetalist, ddthetalist, g,
                                      Ftip, Mlist, Glist, Slist);

  Eigen::VectorXd expected(3);
  expected << 74.69616155, -33.06766016, -3.23057314;

  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(tau(i), expected(i), 1e-6);
  }
}
