#include "WallE/robot_control.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "WallE/trajectory.h"

namespace {

struct ThreeLinkControlData {
  Eigen::VectorXd thetalist;
  Eigen::VectorXd dthetalist;
  Eigen::Vector3d g;
  std::vector<Eigen::MatrixXd> Mlist;
  std::vector<Eigen::MatrixXd> Glist;
  Eigen::Matrix<double, 6, 3> Slist;
};

ThreeLinkControlData MakeThreeLinkControlData() {
  ThreeLinkControlData data;
  data.thetalist.resize(3);
  data.thetalist << 0.1, 0.1, 0.1;

  data.dthetalist.resize(3);
  data.dthetalist << 0.1, 0.2, 0.3;

  data.g = Eigen::Vector3d(0.0, 0.0, -9.8);

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

TEST(RobotControlTest, ComputedTorqueThreeLinkExample) {
  const auto data = MakeThreeLinkControlData();

  Eigen::VectorXd eint(3);
  eint << 0.2, 0.2, 0.2;

  Eigen::VectorXd thetalistd(3);
  thetalistd << 1.0, 1.0, 1.0;

  Eigen::VectorXd dthetalistd(3);
  dthetalistd << 2.0, 1.2, 2.0;

  Eigen::VectorXd ddthetalistd(3);
  ddthetalistd << 0.1, 0.1, 0.1;

  double Kp = 1.3;
  double Ki = 1.2;
  double Kd = 1.1;

  Eigen::VectorXd tau = WallE::RobotControl::ComputedTorque(
      data.thetalist, data.dthetalist, eint, thetalistd, dthetalistd,
      ddthetalistd, data.g, data.Mlist, data.Glist, data.Slist, Kp, Ki, Kd);

  Eigen::VectorXd expected(3);
  expected << 133.00525246, -29.94223324, -3.03276856;
  ASSERT_TRUE(tau.isApprox(expected, 1e-4));
}

TEST(RobotControlTest, ComputedTorqueAlternateGainsFinite) {
  const auto data = MakeThreeLinkControlData();

  Eigen::VectorXd eint = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd thetalistd(3);
  thetalistd << 0.0, 0.0, 0.0;

  Eigen::VectorXd dthetalistd(3);
  dthetalistd << 0.0, 0.0, 0.0;

  Eigen::VectorXd ddthetalistd(3);
  ddthetalistd << 0.0, 0.0, 0.0;

  double Kp = 2.1;
  double Ki = 0.05;
  double Kd = 0.45;

  Eigen::VectorXd tau = WallE::RobotControl::ComputedTorque(
      data.thetalist, data.dthetalist, eint, thetalistd, dthetalistd,
      ddthetalistd, data.g, data.Mlist, data.Glist, data.Slist, Kp, Ki, Kd);

  EXPECT_EQ(tau.size(), 3);
  for (int i = 0; i < tau.size(); ++i) {
    EXPECT_TRUE(std::isfinite(tau(i)));
  }
}

TEST(RobotControlTest, SimulateControlReferenceTrajectory) {
  const auto data = MakeThreeLinkControlData();

  double dt = 0.01;
  Eigen::VectorXd thetaend(3);
  thetaend << M_PI / 2, M_PI / 2, M_PI / 2;
  double Tf = 1.0;
  int N = static_cast<int>(1.0 * Tf / dt);
  int method = 5;

  Eigen::MatrixXd traj =
      WallE::Trajectory::JointTrajectory(data.thetalist, thetaend, Tf, N, method);
  Eigen::MatrixXd thetamatd = traj;
  Eigen::MatrixXd dthetamatd = Eigen::MatrixXd::Zero(N, 3);
  Eigen::MatrixXd ddthetamatd = Eigen::MatrixXd::Zero(N, 3);
  dt = Tf / (N - 1.0);
  for (int i = 0; i < N - 1; ++i) {
    dthetamatd.row(i + 1) = (thetamatd.row(i + 1) - thetamatd.row(i)) / dt;
    ddthetamatd.row(i + 1) = (dthetamatd.row(i + 1) - dthetamatd.row(i)) / dt;
  }

  Eigen::Vector3d gtilde(0.8, 0.2, -8.8);

  std::vector<Eigen::MatrixXd> Mtildelist;
  std::vector<Eigen::MatrixXd> Gtildelist;
  Eigen::Matrix4d Mhat01;
  Mhat01 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0.1,
      0, 0, 0, 1;
  Eigen::Matrix4d Mhat12;
  Mhat12 << 0, 0, 1, 0.3,
      0, 1, 0, 0.2,
      -1, 0, 0, 0,
      0, 0, 0, 1;
  Eigen::Matrix4d Mhat23;
  Mhat23 << 1, 0, 0, 0,
      0, 1, 0, -0.2,
      0, 0, 1, 0.4,
      0, 0, 0, 1;
  Eigen::Matrix4d Mhat34;
  Mhat34 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0.2,
      0, 0, 0, 1;
  Mtildelist = {Mhat01, Mhat12, Mhat23, Mhat34};

  Eigen::VectorXd Ghat1(6);
  Ghat1 << 0.1, 0.1, 0.1, 4, 4, 4;
  Eigen::VectorXd Ghat2(6);
  Ghat2 << 0.3, 0.3, 0.1, 9, 9, 9;
  Eigen::VectorXd Ghat3(6);
  Ghat3 << 0.1, 0.1, 0.1, 3, 3, 3;
  Gtildelist = {Ghat1.asDiagonal(), Ghat2.asDiagonal(), Ghat3.asDiagonal()};

  Eigen::MatrixXd Ftipmat = Eigen::MatrixXd::Ones(N, 6);
  double Kp = 20.0;
  double Ki = 10.0;
  double Kd = 18.0;
  int intRes = 8;

  auto control_traj = WallE::RobotControl::SimulateControl(
      data.thetalist, data.dthetalist, data.g, Ftipmat, data.Mlist, data.Glist,
      data.Slist, thetamatd, dthetamatd, ddthetamatd, gtilde, Mtildelist,
      Gtildelist, Kp, Ki, Kd, dt, intRes);

  ASSERT_EQ(control_traj.size(), 2u);
  Eigen::MatrixXd traj_tau = control_traj.at(0);
  Eigen::MatrixXd traj_theta = control_traj.at(1);

  int mid = static_cast<int>(N / 2) - 1;
  Eigen::MatrixXd result_taumat(3, 3);
  result_taumat << -14.264076502181332, -54.067974287107987,
      -11.265447999884621, 74.439941466730289, -16.78475327873273,
      10.37591119811195, 90.54751171582059, -21.767331772230882,
      2.8921860827244581;

  Eigen::MatrixXd result_thetamat(3, 3);
  result_thetamat << 0.10092028557640544, 0.10190510635161694,
      0.1016066723205018, 0.86520429633566198, 0.82155145166831778,
      0.87396659113443176, 1.5711370254693691, 1.5185661981611596,
      1.5544290589655743;

  Eigen::MatrixXd sampled_tau(3, 3);
  sampled_tau << traj_tau.row(0), traj_tau.row(mid), traj_tau.row(N - 1);
  Eigen::MatrixXd sampled_theta(3, 3);
  sampled_theta << traj_theta.row(0), traj_theta.row(mid),
      traj_theta.row(N - 1);

  ASSERT_TRUE(sampled_tau.isApprox(result_taumat, 1e-4))
      << "sampled_tau:\n" << sampled_tau;
  ASSERT_TRUE(sampled_theta.isApprox(result_thetamat, 1e-4))
      << "sampled_theta:\n" << sampled_theta;
}
