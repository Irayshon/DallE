#include "my_modern_robotics/dynamics.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

TEST(DynamicsTest, MassMatrixExample) {
  Eigen::VectorXd thetalist(3);
  thetalist << 0.1, 0.1, 0.1;

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

  Eigen::MatrixXd mass =
      mymr::Dynamics::MassMatrix(thetalist, Mlist, Glist, Slist);

  Eigen::MatrixXd expected(3, 3);
  expected << 2.25433380e+01, -3.07146754e-01, -7.18426391e-03,
      -3.07146754e-01, 1.96850717e+00, 4.32157368e-01,
      -7.18426391e-03, 4.32157368e-01, 1.91630858e-01;

  for (int r = 0; r < expected.rows(); ++r) {
    for (int c = 0; c < expected.cols(); ++c) {
      EXPECT_NEAR(mass(r, c), expected(r, c), 1e-6);
    }
  }
}
