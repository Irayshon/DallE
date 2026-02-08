#include "WallE/fk.h"

#include "WallE/tools.h"

namespace WallE {
Eigen::MatrixXd FK::FKinBody(const Eigen::MatrixXd& M,
                             const Eigen::MatrixXd& Blist,
                             const Eigen::VectorXd& thetalist) {
  Eigen::MatrixXd T = M;
  int n = static_cast<int>(thetalist.size());
  for (int i = 0; i < n; ++i) {
    T = T * Tools::MatrixExp6(
                Tools::VecTose3(Blist.col(i) * thetalist(i)));
  }
  return T;
}

Eigen::MatrixXd FK::FKinSpace(const Eigen::MatrixXd& M,
                              const Eigen::MatrixXd& Slist,
                              const Eigen::VectorXd& thetalist) {
  Eigen::MatrixXd T = M;
  int n = static_cast<int>(thetalist.size());
  for (int i = n - 1; i >= 0; --i) {
    T = Tools::MatrixExp6(Tools::VecTose3(Slist.col(i) * thetalist(i))) * T;
  }
  return T;
}
}  // namespace WallE
