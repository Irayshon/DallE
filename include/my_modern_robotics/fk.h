#pragma once

#include <Eigen/Dense>

namespace mymr {
/**
 * @brief Forward kinematics functions.
 */
class FK {
 public:
  /**
   * @brief Compute forward kinematics in the body frame.
   * @param M Home configuration of the end-effector.
   * @param Blist 6xn screw axes in the body frame.
   * @param thetalist n-vector of joint angles.
   * @return End-effector transform in SE(3).
   */
  static Eigen::MatrixXd FKinBody(const Eigen::MatrixXd& M,
                                  const Eigen::MatrixXd& Blist,
                                  const Eigen::VectorXd& thetalist);

  /**
   * @brief Compute forward kinematics in the space frame.
   * @param M Home configuration of the end-effector.
   * @param Slist 6xn screw axes in the space frame.
   * @param thetalist n-vector of joint angles.
   * @return End-effector transform in SE(3).
   */
  static Eigen::MatrixXd FKinSpace(const Eigen::MatrixXd& M,
                                   const Eigen::MatrixXd& Slist,
                                   const Eigen::VectorXd& thetalist);
};
}  // namespace mymr
