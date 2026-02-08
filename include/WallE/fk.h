#pragma once

#include <Eigen/Dense>

namespace WallE {
/**
 * @brief Forward kinematics functions.
 */
class FK {
 public:
  /**
   * @brief Compute forward kinematics in the body frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param Blist Eigen::MatrixXd 6xn screw axes in the body frame.
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @return Eigen::MatrixXd end-effector transform in SE(3).
   */
  static Eigen::MatrixXd FKinBody(const Eigen::MatrixXd& M,
                                  const Eigen::MatrixXd& Blist,
                                  const Eigen::VectorXd& thetalist);

  /**
   * @brief Compute forward kinematics in the space frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @return Eigen::MatrixXd end-effector transform in SE(3).
   */
  static Eigen::MatrixXd FKinSpace(const Eigen::MatrixXd& M,
                                   const Eigen::MatrixXd& Slist,
                                   const Eigen::VectorXd& thetalist);
};
}  // namespace WallE
