#include "WallE/inverse_dynamics.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

namespace {

struct ThreeLinkExampleData {
  Eigen::VectorXd thetalist;
  Eigen::VectorXd dthetalist;
  Eigen::VectorXd ddthetalist;
  Eigen::Vector3d g;
  Eigen::VectorXd Ftip;
  std::vector<Eigen::MatrixXd> Mlist;
  std::vector<Eigen::MatrixXd> Glist;
  Eigen::Matrix<double, 6, 3> Slist;
};

ThreeLinkExampleData MakeThreeLinkExample() {
  ThreeLinkExampleData data;
  data.thetalist.resize(3);
  data.thetalist << 0.1, 0.1, 0.1;

  data.dthetalist.resize(3);
  data.dthetalist << 0.1, 0.2, 0.3;

  data.ddthetalist.resize(3);
  data.ddthetalist << 2.0, 1.5, 1.0;

  data.g = Eigen::Vector3d(0.0, 0.0, -9.8);

  data.Ftip.resize(6);
  data.Ftip << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

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

  data.Mlist = {M01, M12, M23, M34};

  Eigen::Matrix<double, 6, 6> G1 = Eigen::Matrix<double, 6, 6>::Zero();
  G1.diagonal() << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;

  Eigen::Matrix<double, 6, 6> G2 = Eigen::Matrix<double, 6, 6>::Zero();
  G2.diagonal() << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;

  Eigen::Matrix<double, 6, 6> G3 = Eigen::Matrix<double, 6, 6>::Zero();
  G3.diagonal() << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

  data.Glist = {G1, G2, G3};

  data.Slist << 1, 0, 0,
      0, 1, 1,
      1, 0, 0,
      0, -0.089, -0.089,
      1, 0, 0,
      0, 0, 0.425;

  return data;
}

}  // namespace

TEST(InverseDynamicsTest, ThreeLinkExample) {
  const auto data = MakeThreeLinkExample();

  Eigen::VectorXd tau =
      WallE::InverseDynamics::Compute(data.thetalist, data.dthetalist,
                                      data.ddthetalist, data.g, data.Ftip,
                                      data.Mlist, data.Glist, data.Slist);

  Eigen::VectorXd expected(3);
  expected << 74.69616155, -33.06766016, -3.23057314;

  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(tau(i), expected(i), 1e-6);
  }
}

TEST(InverseDynamicsTest, GravityOnlyExample) {
  const auto data = MakeThreeLinkExample();
  Eigen::VectorXd zeros = Eigen::VectorXd::Zero(data.thetalist.size());
  Eigen::VectorXd Ftip = Eigen::VectorXd::Zero(6);

  Eigen::VectorXd tau = WallE::InverseDynamics::Compute(
      data.thetalist, zeros, zeros, data.g, Ftip, data.Mlist, data.Glist,
      data.Slist);

  Eigen::VectorXd expected(3);
  expected << 28.40331262, -37.64094817, -5.4415892;

  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(tau(i), expected(i), 1e-6);
  }
}
