#include "WallE/robot_control.h"

#include "WallE/dynamics.h"
#include "WallE/inverse_dynamics.h"

namespace WallE {
Eigen::VectorXd RobotControl::ComputedTorque(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const Eigen::VectorXd& eint,
    const Eigen::VectorXd& thetalistd,
    const Eigen::VectorXd& dthetalistd,
    const Eigen::VectorXd& ddthetalistd,
    const Eigen::Vector3d& g,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist,
    double Kp,
    double Ki,
    double Kd) {
  Eigen::VectorXd e = thetalistd - thetalist;
  Eigen::VectorXd edot = dthetalistd - dthetalist;
  Eigen::VectorXd tau_feedforward =
      Dynamics::MassMatrix(thetalist, Mlist, Glist, Slist) *
      (Kp * e + Ki * (eint + e) + Kd * edot);
  Eigen::VectorXd Ftip = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd tau_inversedyn = InverseDynamics::Compute(
      thetalist, dthetalist, ddthetalistd, g, Ftip, Mlist, Glist, Slist);
  return tau_feedforward + tau_inversedyn;
}

std::vector<Eigen::MatrixXd> RobotControl::SimulateControl(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const Eigen::Vector3d& g,
    const Eigen::MatrixXd& Ftipmat,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist,
    const Eigen::MatrixXd& thetamatd,
    const Eigen::MatrixXd& dthetamatd,
    const Eigen::MatrixXd& ddthetamatd,
    const Eigen::Vector3d& gtilde,
    const std::vector<Eigen::MatrixXd>& Mtildelist,
    const std::vector<Eigen::MatrixXd>& Gtildelist,
    double Kp,
    double Ki,
    double Kd,
    double dt,
    int intRes) {
  Eigen::MatrixXd FtipmatT = Ftipmat.transpose();
  Eigen::MatrixXd thetamatdT = thetamatd.transpose();
  Eigen::MatrixXd dthetamatdT = dthetamatd.transpose();
  Eigen::MatrixXd ddthetamatdT = ddthetamatd.transpose();
  int m = static_cast<int>(thetamatdT.rows());
  int n = static_cast<int>(thetamatdT.cols());
  Eigen::VectorXd thetacurrent = thetalist;
  Eigen::VectorXd dthetacurrent = dthetalist;
  Eigen::VectorXd eint = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd taumatT = Eigen::MatrixXd::Zero(m, n);
  Eigen::MatrixXd thetamatT = Eigen::MatrixXd::Zero(m, n);

  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd taulist = RobotControl::ComputedTorque(
        thetacurrent, dthetacurrent, eint, thetamatdT.col(i),
        dthetamatdT.col(i), ddthetamatdT.col(i), gtilde, Mtildelist, Gtildelist,
        Slist, Kp, Ki, Kd);
    for (int j = 0; j < intRes; ++j) {
      Eigen::VectorXd ddthetalist = Dynamics::ForwardDynamics(
          thetacurrent, dthetacurrent, taulist, g, FtipmatT.col(i), Mlist,
          Glist, Slist);
      auto next = Dynamics::EulerStep(thetacurrent, dthetacurrent, ddthetalist,
                                      dt / static_cast<double>(intRes));
      thetacurrent = next.at(0);
      dthetacurrent = next.at(1);
    }
    taumatT.col(i) = taulist;
    thetamatT.col(i) = thetacurrent;
    eint += dt * (thetamatdT.col(i) - thetacurrent);
  }

  std::vector<Eigen::MatrixXd> control_traj;
  control_traj.reserve(2);
  control_traj.push_back(taumatT.transpose());
  control_traj.push_back(thetamatT.transpose());
  return control_traj;
}
}  // namespace WallE
