#include "WallE/trajectory.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {
void ExpectMatrixNear(const Eigen::MatrixXd& actual,
                      const Eigen::MatrixXd& expected,
                      double tol) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (int r = 0; r < expected.rows(); ++r) {
    for (int c = 0; c < expected.cols(); ++c) {
      EXPECT_NEAR(actual(r, c), expected(r, c), tol);
    }
  }
}
}  // namespace

TEST(TrajectoryTest, QuinticTimeScalingExpectedValues) {
  double Tf = 4.0;
  EXPECT_NEAR(WallE::Trajectory::QuinticTimeScaling(Tf, 0.0), 0.0, 1e-12);
  EXPECT_NEAR(WallE::Trajectory::QuinticTimeScaling(Tf, Tf / 4.0), 0.103515625,
              1e-12);
  EXPECT_NEAR(WallE::Trajectory::QuinticTimeScaling(Tf, Tf / 2.0), 0.5, 1e-12);
  EXPECT_NEAR(WallE::Trajectory::QuinticTimeScaling(Tf, Tf), 1.0, 1e-12);
}

TEST(TrajectoryTest, ScrewTrajectoryEndpoints) {
  Eigen::Matrix4d Xstart = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Xend = Eigen::Matrix4d::Identity();
  Xend(0, 0) = 0.0;
  Xend(0, 1) = -1.0;
  Xend(1, 0) = 1.0;
  Xend(1, 1) = 0.0;
  Xend(0, 3) = 1.0;
  Xend(1, 3) = 2.0;
  Xend(2, 3) = 3.0;

  double Tf = 2.0;
  int N = 5;
  int method = 5;

  std::vector<Eigen::MatrixXd> traj =
      WallE::Trajectory::ScrewTrajectory(Xstart, Xend, Tf, N, method);

  ASSERT_EQ(static_cast<int>(traj.size()), N);
  ExpectMatrixNear(traj.front(), Xstart, 1e-6);
  ExpectMatrixNear(traj.back(), Xend, 1e-6);
}

TEST(TrajectoryTest, CartesianTrajectoryEndpoints) {
  Eigen::Matrix4d Xstart = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Xend = Eigen::Matrix4d::Identity();
  Xend(0, 0) = 0.0;
  Xend(0, 1) = -1.0;
  Xend(1, 0) = 1.0;
  Xend(1, 1) = 0.0;
  Xend(0, 3) = -0.5;
  Xend(1, 3) = 0.25;
  Xend(2, 3) = 1.5;

  double Tf = 2.0;
  int N = 5;
  int method = 5;

  std::vector<Eigen::MatrixXd> traj =
      WallE::Trajectory::CartesianTrajectory(Xstart, Xend, Tf, N, method);

  ASSERT_EQ(static_cast<int>(traj.size()), N);
  ExpectMatrixNear(traj.front(), Xstart, 1e-6);
  ExpectMatrixNear(traj.back(), Xend, 1e-6);
}

TEST(TrajectoryTest, JointTrajectoryCubicExample) {
  Eigen::VectorXd thetastart(8);
  thetastart << 1, 0, 0, 1, 1, 0.2, 0, 1;
  Eigen::VectorXd thetaend(8);
  thetaend << 1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1;
  double Tf = 4.0;
  int N = 6;
  int method = 3;

  Eigen::MatrixXd traj =
      WallE::Trajectory::JointTrajectory(thetastart, thetaend, Tf, N, method);

  Eigen::Matrix<double, 6, 8> expected;
  expected << 1, 0, 0, 1, 1, 0.2, 0, 1,
      1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1,
      1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1,
      1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1,
      1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1,
      1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1;

  ExpectMatrixNear(traj, expected, 1e-6);
}
