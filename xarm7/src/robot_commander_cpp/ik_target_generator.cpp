#include <WallE/fk.h>
#include <WallE/ik.h>

#include <Eigen/Dense>

#include <iostream>

int main() {
  Eigen::Matrix<double, 6, 7> Slist;
  Slist << 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, -1, 0, -1, 0,
      1, 0, 1, 0, 1, 0, -1,
      0, -0.333, 0, 0.649, 0, 1.033, 0,
      0, 0, 0, 0, 0, 0, 0.088,
      0, 0, 0, -0.0825, 0, 0, 0;

  Eigen::Matrix4d M;
  M << 1, 0, 0, 0.088,
      0, 1, 0, 0,
      0, 0, 1, 1.033,
      0, 0, 0, 1;

  Eigen::VectorXd ref(7);
  ref << 0.3, -0.4, 0.25, -0.35, 0.2, 0.35, -0.2;
  Eigen::Matrix4d T = WallE::FK::FKinSpace(M, Slist, ref);

  Eigen::VectorXd guess = Eigen::VectorXd::Zero(7);
  bool ok = WallE::IK::IKinSpace(Slist, M, T, guess, 1e-3, 1e-3);

  std::cout << "ok=" << (ok ? 1 : 0) << " joints=";
  for (int i = 0; i < guess.size(); ++i) {
    std::cout << (i == 0 ? "" : ",") << guess(i);
  }
  std::cout << "\n";
  return ok ? 0 : 1;
}
