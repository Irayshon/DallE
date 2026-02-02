#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
class InverseDynamics {
 public:
  static Eigen::VectorXd Compute(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& ddthetalist,
      const Eigen::Vector3d& g,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);
};
}  // namespace mymr
