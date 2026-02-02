#include "my_modern_robotics/trajectory.h"

#include "my_modern_robotics/tools.h"

namespace mymr {
double Trajectory::CubicTimeScaling(double Tf, double t) {
  double ratio = t / Tf;
  return 3.0 * ratio * ratio - 2.0 * ratio * ratio * ratio;
}

double Trajectory::QuinticTimeScaling(double Tf, double t) {
  double ratio = t / Tf;
  double ratio2 = ratio * ratio;
  double ratio3 = ratio2 * ratio;
  double ratio4 = ratio3 * ratio;
  double ratio5 = ratio4 * ratio;
  return 10.0 * ratio3 - 15.0 * ratio4 + 6.0 * ratio5;
}

Eigen::MatrixXd Trajectory::JointTrajectory(const Eigen::VectorXd& thetastart,
                                            const Eigen::VectorXd& thetaend,
                                            double Tf,
                                            int N,
                                            int method) {
  double timegap = Tf / (static_cast<double>(N) - 1.0);
  Eigen::MatrixXd traj(N, thetastart.size());
  for (int i = 0; i < N; ++i) {
    double s = (method == 3)
                   ? CubicTimeScaling(Tf, timegap * static_cast<double>(i))
                   : QuinticTimeScaling(Tf, timegap * static_cast<double>(i));
    traj.row(i) = (s * thetaend + (1.0 - s) * thetastart).transpose();
  }
  return traj;
}

std::vector<Eigen::MatrixXd> Trajectory::ScrewTrajectory(
    const Eigen::MatrixXd& Xstart,
    const Eigen::MatrixXd& Xend,
    double Tf,
    int N,
    int method) {
  double timegap = Tf / (static_cast<double>(N) - 1.0);
  std::vector<Eigen::MatrixXd> traj(N);
  Eigen::MatrixXd log_se3 =
      Tools::MatrixLog6(Tools::TransInv(Xstart) * Xend);
  for (int i = 0; i < N; ++i) {
    double s = (method == 3)
                   ? CubicTimeScaling(Tf, timegap * static_cast<double>(i))
                   : QuinticTimeScaling(Tf, timegap * static_cast<double>(i));
    traj.at(i) = Xstart * Tools::MatrixExp6(log_se3 * s);
  }
  return traj;
}

std::vector<Eigen::MatrixXd> Trajectory::CartesianTrajectory(
    const Eigen::MatrixXd& Xstart,
    const Eigen::MatrixXd& Xend,
    double Tf,
    int N,
    int method) {
  double timegap = Tf / (static_cast<double>(N) - 1.0);
  std::vector<Eigen::MatrixXd> traj(N);
  auto rp_start = Tools::TransToRp(Xstart);
  auto rp_end = Tools::TransToRp(Xend);
  Eigen::Matrix3d Rstart = rp_start.at(0);
  Eigen::Vector3d pstart = rp_start.at(1);
  Eigen::Matrix3d Rend = rp_end.at(0);
  Eigen::Vector3d pend = rp_end.at(1);
  Eigen::Matrix3d log_rot = Tools::MatrixLog3(Rstart.transpose() * Rend);
  for (int i = 0; i < N; ++i) {
    double s = (method == 3)
                   ? CubicTimeScaling(Tf, timegap * static_cast<double>(i))
                   : QuinticTimeScaling(Tf, timegap * static_cast<double>(i));
    Eigen::Matrix3d R = Rstart * Tools::MatrixExp3(log_rot * s);
    Eigen::Vector3d p = s * pend + (1.0 - s) * pstart;
    traj.at(i) = Tools::RpToTrans(R, p);
  }
  return traj;
}
}  // namespace mymr
