#pragma once

#include <Eigen/Dense>

namespace mymr {
/**
 * @brief Inverse kinematics functions.
 */
class IK {
 public:
  /**
   * @brief Compute inverse kinematics in the body frame.
   * @param Blist 6xn screw axes in the body frame.
   * @param M Home configuration of the end-effector.
   * @param T Desired end-effector configuration.
   * @param thetalist Initial guess; updated with solution on success.
   * @param eomg Orientation error tolerance.
   * @param ev Position error tolerance.
   * @return True if solution converged within tolerances.
   */
  static bool IKinBody(const Eigen::MatrixXd& Blist,
                       const Eigen::MatrixXd& M,
                       const Eigen::MatrixXd& T,
                       Eigen::VectorXd& thetalist,
                       double eomg,
                       double ev);

  /**
   * @brief Compute inverse kinematics in the space frame.
   * @param Slist 6xn screw axes in the space frame.
   * @param M Home configuration of the end-effector.
   * @param T Desired end-effector configuration.
   * @param thetalist Initial guess; updated with solution on success.
   * @param eomg Orientation error tolerance.
   * @param ev Position error tolerance.
   * @return True if solution converged within tolerances.
   */
  static bool IKinSpace(const Eigen::MatrixXd& Slist,
                        const Eigen::MatrixXd& M,
                        const Eigen::MatrixXd& T,
                        Eigen::VectorXd& thetalist,
                        double eomg,
                        double ev);
};
}  // namespace mymr
