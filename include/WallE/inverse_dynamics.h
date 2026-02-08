#pragma once

#include <Eigen/Dense>
#include <vector>

namespace WallE {
/**
 * @brief Inverse dynamics solver.
 */
class InverseDynamics {
 public:
  /**
   * @brief Compute joint torques from desired motion and external forces.
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param dthetalist Eigen::VectorXd n-vector of joint velocities.
   * @param ddthetalist Eigen::VectorXd n-vector of joint accelerations.
   * @param g Eigen::Vector3d gravity vector.
   * @param Ftip Eigen::VectorXd spatial force at the end-effector.
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::VectorXd n-vector of joint forces/torques.
   */
  static Eigen::VectorXd Compute(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& ddthetalist,
      const Eigen::Vector3d& g,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);
};
}  // namespace WallE
