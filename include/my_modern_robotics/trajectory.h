#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
/**
 * @brief Time scaling and trajectory generation utilities.
 */
class Trajectory {
 public:
  /**
   * @brief Cubic time scaling s(t).
   * @param Tf Total duration.
   * @param t Current time.
   * @return Scalar s in [0,1].
   */
  static double CubicTimeScaling(double Tf, double t);

  /**
   * @brief Quintic time scaling s(t).
   * @param Tf Total duration.
   * @param t Current time.
   * @return Scalar s in [0,1].
   */
  static double QuinticTimeScaling(double Tf, double t);

  /**
   * @brief Generate a joint-space trajectory.
   * @param thetastart Start joint angles.
   * @param thetaend End joint angles.
   * @param Tf Total duration.
   * @param N Number of points.
   * @param method Time scaling method (3 or 5).
   * @return N x n matrix of joint positions.
   */
  static Eigen::MatrixXd JointTrajectory(const Eigen::VectorXd& thetastart,
                                         const Eigen::VectorXd& thetaend,
                                         double Tf,
                                         int N,
                                         int method);

  /**
   * @brief Generate a screw trajectory in SE(3).
   * @param Xstart Start end-effector configuration.
   * @param Xend End end-effector configuration.
   * @param Tf Total duration.
   * @param N Number of points.
   * @param method Time scaling method (3 or 5).
   * @return Vector of SE(3) matrices.
   */
  static std::vector<Eigen::MatrixXd> ScrewTrajectory(
      const Eigen::MatrixXd& Xstart,
      const Eigen::MatrixXd& Xend,
      double Tf,
      int N,
      int method);

  /**
   * @brief Generate a Cartesian trajectory in SE(3).
   * @param Xstart Start end-effector configuration.
   * @param Xend End end-effector configuration.
   * @param Tf Total duration.
   * @param N Number of points.
   * @param method Time scaling method (3 or 5).
   * @return Vector of SE(3) matrices.
   */
  static std::vector<Eigen::MatrixXd> CartesianTrajectory(
      const Eigen::MatrixXd& Xstart,
      const Eigen::MatrixXd& Xend,
      double Tf,
      int N,
      int method);
};
}  // namespace mymr
