#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
class RobotControl {
 public:
  static Eigen::VectorXd ComputedTorque(
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
      double Kd);

  static std::vector<Eigen::MatrixXd> SimulateControl(
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
      int intRes);
};
}  // namespace mymr
