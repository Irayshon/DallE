#include "WallE/inverse_dynamics.h"

#include "WallE/tools.h"

#include <vector>

namespace WallE {
Eigen::VectorXd InverseDynamics::Compute(
    const Eigen::VectorXd& thetalist,
    const Eigen::VectorXd& dthetalist,
    const Eigen::VectorXd& ddthetalist,
    const Eigen::Vector3d& g,
    const Eigen::VectorXd& Ftip,
    const std::vector<Eigen::MatrixXd>& Mlist,
    const std::vector<Eigen::MatrixXd>& Glist,
    const Eigen::MatrixXd& Slist) {
  int n = static_cast<int>(thetalist.size());
  Eigen::MatrixXd Mi = Eigen::MatrixXd::Identity(4, 4);
  Eigen::MatrixXd Ai = Eigen::MatrixXd::Zero(6, n);
  std::vector<Eigen::MatrixXd> AdTi(n + 1);
  Eigen::MatrixXd Vi = Eigen::MatrixXd::Zero(6, n + 1);
  Eigen::MatrixXd Vdi = Eigen::MatrixXd::Zero(6, n + 1);
  Vdi.col(0) << 0.0, 0.0, 0.0, -g(0), -g(1), -g(2);
  AdTi[n] = Tools::Adjoint(Tools::TransInv(Mlist.at(n)));
  Eigen::VectorXd Fi = Ftip;
  Eigen::VectorXd taulist = Eigen::VectorXd::Zero(n);

  for (int i = 0; i < n; ++i) {
    Mi = Mi * Mlist.at(i);
    Ai.col(i) = Tools::Adjoint(Tools::TransInv(Mi)) * Slist.col(i);
    AdTi[i] = Tools::Adjoint(
        Tools::MatrixExp6(Tools::VecTose3(Ai.col(i) * -thetalist(i))) *
        Tools::TransInv(Mlist.at(i)));
    Vi.col(i + 1) = AdTi[i] * Vi.col(i) + Ai.col(i) * dthetalist(i);
    Vdi.col(i + 1) = AdTi[i] * Vdi.col(i) + Ai.col(i) * ddthetalist(i) +
                     Tools::ad(Vi.col(i + 1)) * Ai.col(i) * dthetalist(i);
  }

  for (int i = n - 1; i >= 0; --i) {
    Fi = AdTi[i + 1].transpose() * Fi + Glist.at(i) * Vdi.col(i + 1) -
         Tools::ad(Vi.col(i + 1)).transpose() *
             (Glist.at(i) * Vi.col(i + 1));
    taulist(i) = Fi.transpose() * Ai.col(i);
  }

  return taulist;
}
}  // namespace WallE
