#include <WallE/fk.h>
#include <WallE/ik.h>

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  constexpr int kPoints = 40;
  constexpr double kDt = 0.6;

  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 1, 0, 0,
      0, 1, 1,
      0, 0, 0,
      0, 0, 0.2,
      0, 0, 0,
      0, -0.08, -0.08;

  Eigen::Matrix4d M;
  M << 1, 0, 0, -0.08,
      0, 1, 0, 0,
      0, 0, 1, -0.213,
      0, 0, 0, 1;

  Eigen::Vector3d theta_guess(0.0, 0.8, -1.5);
  std::ofstream out("/tmp/quadruped_ik_long.csv", std::ios::trunc);
  if (!out.is_open()) {
    return 1;
  }

  out << "t,hip,thigh,calf,ok\n";

  for (int i = 0; i < kPoints; ++i) {
    double s = (2.0 * M_PI * static_cast<double>(i)) / static_cast<double>(kPoints - 1);
    Eigen::Vector3d theta_ref;
    theta_ref(0) = 0.18 * std::sin(s);
    theta_ref(1) = 0.80 + 0.15 * std::sin(s);
    theta_ref(2) = -1.50 + 0.22 * std::cos(s);

    Eigen::Matrix4d T_target = WallE::FK::FKinSpace(M, Slist, theta_ref);

    Eigen::VectorXd theta_solve = theta_guess;
    bool ok = WallE::IK::IKinSpace(Slist, M, T_target, theta_solve, 1e-4, 1e-4);
    if (!ok) {
      out << (i * kDt) << ',' << theta_solve(0) << ',' << theta_solve(1) << ',' << theta_solve(2) << ",0\n";
      return 2;
    }
    theta_guess = theta_solve;
    out << (i * kDt) << ',' << theta_solve(0) << ',' << theta_solve(1) << ',' << theta_solve(2) << ",1\n";
  }

  std::cout << "generated_points=" << kPoints << " total_time=" << ((kPoints - 1) * kDt) << "\n";
  return 0;
}
