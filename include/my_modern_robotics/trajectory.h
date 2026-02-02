#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
class Trajectory {
 public:
  static double CubicTimeScaling(double Tf, double t);
  static double QuinticTimeScaling(double Tf, double t);
  static Eigen::MatrixXd JointTrajectory(const Eigen::VectorXd& thetastart,
                                         const Eigen::VectorXd& thetaend,
                                         double Tf,
                                         int N,
                                         int method);
  static std::vector<Eigen::MatrixXd> ScrewTrajectory(
      const Eigen::MatrixXd& Xstart,
      const Eigen::MatrixXd& Xend,
      double Tf,
      int N,
      int method);
  static std::vector<Eigen::MatrixXd> CartesianTrajectory(
      const Eigen::MatrixXd& Xstart,
      const Eigen::MatrixXd& Xend,
      double Tf,
      int N,
      int method);
};
}  // namespace mymr
