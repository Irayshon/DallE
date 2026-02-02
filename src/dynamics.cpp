#include "my_modern_robotics/dynamics.h"

#include "my_modern_robotics/inverse_dynamics.h"

#include <vector>

namespace mymr {
Eigen::MatrixXd Dynamics::MassMatrix(
    const Eigen::VectorXd& thetalist,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  int n = static_cast<int>(thetalist.size());
  Eigen::MatrixXd mass_matrix(n, n);
  Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n);
  Eigen::Vector3d zero_g = Eigen::Vector3d::Zero();
  Eigen::VectorXd zero_f = Eigen::VectorXd::Zero(6);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ddthetalist = Eigen::VectorXd::Zero(n);
    ddthetalist(i) = 1.0;
    mass_matrix.col(i) = InverseDynamics::Compute(
        thetalist, zeros, ddthetalist, zero_g, zero_f, Mlist, Glist, Slist);
  }
  return mass_matrix;
}

Eigen::VectorXd Dynamics::VelQuadraticForces(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  int n = static_cast<int>(thetalist.size());
  return InverseDynamics::Compute(thetalist, dthetalist,
                                  Eigen::VectorXd::Zero(n),
                                  Eigen::Vector3d::Zero(),
                                  Eigen::VectorXd::Zero(6), Mlist, Glist,
                                  Slist);
}

Eigen::VectorXd Dynamics::GravityForces(
    const Eigen::VectorXd& thetalist,
    const Eigen::Vector3d& g,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  int n = static_cast<int>(thetalist.size());
  return InverseDynamics::Compute(thetalist, Eigen::VectorXd::Zero(n),
                                  Eigen::VectorXd::Zero(n), g,
                                  Eigen::VectorXd::Zero(6), Mlist, Glist,
                                  Slist);
}

Eigen::VectorXd Dynamics::EndEffectorForces(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& Ftip,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  int n = static_cast<int>(thetalist.size());
  return InverseDynamics::Compute(thetalist, Eigen::VectorXd::Zero(n),
                                  Eigen::VectorXd::Zero(n),
                                  Eigen::Vector3d::Zero(), Ftip, Mlist, Glist,
                                  Slist);
}

Eigen::VectorXd Dynamics::ForwardDynamics(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const Eigen::VectorXd& taulist,
    const Eigen::Vector3d& g,
    const Eigen::VectorXd& Ftip,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  Eigen::MatrixXd mass_matrix = MassMatrix(thetalist, Mlist, Glist, Slist);
  Eigen::VectorXd rhs = taulist - VelQuadraticForces(thetalist, dthetalist,
                                                     Mlist, Glist, Slist) -
                        GravityForces(thetalist, g, Mlist, Glist, Slist) -
                        EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist);
  return mass_matrix.ldlt().solve(rhs);
}

std::vector<Eigen::VectorXd> Dynamics::EulerStep(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const Eigen::VectorXd& ddthetalist,
    double dt) {
  std::vector<Eigen::VectorXd> next;
  next.reserve(2);
  next.push_back(thetalist + dt * dthetalist);
  next.push_back(dthetalist + dt * ddthetalist);
  return next;
}

Eigen::MatrixXd Dynamics::InverseDynamicsTrajectory(
    const Eigen::MatrixXd& thetamat,
    const Eigen::MatrixXd& dthetamat,
    const Eigen::MatrixXd& ddthetamat,
    const Eigen::Vector3d& g,
    const Eigen::MatrixXd& Ftipmat,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  int steps = static_cast<int>(thetamat.rows());
  int n = static_cast<int>(thetamat.cols());
  Eigen::MatrixXd taumat(steps, n);
  for (int i = 0; i < steps; ++i) {
    Eigen::VectorXd thetalist = thetamat.row(i).transpose();
    Eigen::VectorXd dthetalist = dthetamat.row(i).transpose();
    Eigen::VectorXd ddthetalist = ddthetamat.row(i).transpose();
    Eigen::VectorXd Ftip = Eigen::VectorXd::Zero(6);
    if (Ftipmat.size() != 0) {
      Ftip = Ftipmat.row(i).transpose();
    }
    taumat.row(i) = InverseDynamics::Compute(thetalist, dthetalist, ddthetalist,
                                             g, Ftip, Mlist, Glist, Slist)
                        .transpose();
  }
  return taumat;
}

std::vector<Eigen::MatrixXd> Dynamics::ForwardDynamicsTrajectory(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const Eigen::MatrixXd& taumat,
    const Eigen::Vector3d& g,
    const Eigen::MatrixXd& Ftipmat,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist,
    double dt,
    int intRes) {
  int steps = static_cast<int>(taumat.rows());
  int n = static_cast<int>(taumat.cols());
  Eigen::MatrixXd thetamat(steps, n);
  Eigen::MatrixXd dthetamat(steps, n);
  Eigen::VectorXd theta = thetalist;
  Eigen::VectorXd dtheta = dthetalist;
  thetamat.row(0) = theta.transpose();
  dthetamat.row(0) = dtheta.transpose();
  for (int i = 0; i < steps - 1; ++i) {
    Eigen::VectorXd Ftip = Eigen::VectorXd::Zero(6);
    if (Ftipmat.size() != 0) {
      Ftip = Ftipmat.row(i).transpose();
    }
    for (int j = 0; j < intRes; ++j) {
      Eigen::VectorXd ddthetalist = ForwardDynamics(
          theta, dtheta, taumat.row(i).transpose(), g, Ftip, Mlist, Glist,
          Slist);
      auto next = EulerStep(theta, dtheta, ddthetalist, dt / intRes);
      theta = next.at(0);
      dtheta = next.at(1);
    }
    thetamat.row(i + 1) = theta.transpose();
    dthetamat.row(i + 1) = dtheta.transpose();
  }
  return {thetamat, dthetamat};
}
}  // namespace mymr
