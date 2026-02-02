#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
/**
 * @brief Inverse dynamics solver.
 */
class InverseDynamics {
 public:
  /**
   * @brief Compute joint torques from desired motion and external forces.
   * @param thetalist n-vector of joint angles.
   * @param dthetalist n-vector of joint velocities.
   * @param ddthetalist n-vector of joint accelerations.
   * @param g Gravity vector.
   * @param Ftip Spatial force applied at the end-effector.
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return n-vector of joint forces/torques.
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
}  // namespace mymr
