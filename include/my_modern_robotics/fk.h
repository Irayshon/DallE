#pragma once

#include <Eigen/Dense>

namespace mymr {
class FK {
 public:
  static Eigen::MatrixXd FKinBody(const Eigen::MatrixXd& M,
                                  const Eigen::MatrixXd& Blist,
                                  const Eigen::VectorXd& thetalist);
  static Eigen::MatrixXd FKinSpace(const Eigen::MatrixXd& M,
                                   const Eigen::MatrixXd& Slist,
                                   const Eigen::VectorXd& thetalist);
};
}  // namespace mymr
