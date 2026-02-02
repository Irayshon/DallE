#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
class Dynamics {
 public:
  static Eigen::MatrixXd MassMatrix(
      const Eigen::VectorXd& thetalist,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  static Eigen::VectorXd VelQuadraticForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  static Eigen::VectorXd GravityForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::Vector3d& g,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  static Eigen::VectorXd EndEffectorForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  static Eigen::VectorXd ForwardDynamics(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& taulist,
      const Eigen::Vector3d& g,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  static std::vector<Eigen::VectorXd> EulerStep(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& ddthetalist,
      double dt);

  static Eigen::MatrixXd InverseDynamicsTrajectory(
      const Eigen::MatrixXd& thetamat,
      const Eigen::MatrixXd& dthetamat,
      const Eigen::MatrixXd& ddthetamat,
      const Eigen::Vector3d& g,
      const Eigen::MatrixXd& Ftipmat,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  static std::vector<Eigen::MatrixXd> ForwardDynamicsTrajectory(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::MatrixXd& taumat,
      const Eigen::Vector3d& g,
      const Eigen::MatrixXd& Ftipmat,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist,
      double dt,
      int intRes);
};
}  // namespace mymr
