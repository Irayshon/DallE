#pragma once

#include <Eigen/Dense>
#include <vector>

namespace WallE {
/**
 * @brief Time scaling and trajectory generation utilities.
 */
class Trajectory {
 public:
  /**
   * @brief Cubic time scaling s(t).
   * @param Tf double total duration.
   * @param t double current time.
   * @return double scalar s in [0,1].
   */
  static double CubicTimeScaling(double Tf, double t);

  /**
   * @brief Quintic time scaling s(t).
   * @param Tf double total duration.
   * @param t double current time.
   * @return double scalar s in [0,1].
   */
  static double QuinticTimeScaling(double Tf, double t);

  /**
   * @brief Generate a joint-space trajectory.
   * @param thetastart Eigen::VectorXd start joint angles.
   * @param thetaend Eigen::VectorXd end joint angles.
   * @param Tf double total duration.
   * @param N int number of points.
   * @param method int time scaling method (3 or 5).
   * @return Eigen::MatrixXd N x n matrix of joint positions.
   */
  static Eigen::MatrixXd JointTrajectory(const Eigen::VectorXd& thetastart,
                                         const Eigen::VectorXd& thetaend,
                                         double Tf,
                                         int N,
                                         int method);

  /**
   * @brief Generate a screw trajectory in SE(3).
   * @param Xstart Eigen::MatrixXd start end-effector configuration.
   * @param Xend Eigen::MatrixXd end end-effector configuration.
   * @param Tf double total duration.
   * @param N int number of points.
   * @param method int time scaling method (3 or 5).
   * @return std::vector<Eigen::MatrixXd> SE(3) matrices.
   */
  static std::vector<Eigen::MatrixXd> ScrewTrajectory(
      const Eigen::MatrixXd& Xstart,
      const Eigen::MatrixXd& Xend,
      double Tf,
      int N,
      int method);

  /**
   * @brief Generate a Cartesian trajectory in SE(3).
   * @param Xstart Eigen::MatrixXd start end-effector configuration.
   * @param Xend Eigen::MatrixXd end end-effector configuration.
   * @param Tf double total duration.
   * @param N int number of points.
   * @param method int time scaling method (3 or 5).
   * @return std::vector<Eigen::MatrixXd> SE(3) matrices.
   */
  static std::vector<Eigen::MatrixXd> CartesianTrajectory(
      const Eigen::MatrixXd& Xstart,
      const Eigen::MatrixXd& Xend,
      double Tf,
      int N,
      int method);
};
}  // namespace WallE
