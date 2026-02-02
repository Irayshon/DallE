#pragma once

#include <Eigen/Dense>

namespace mymr {
class IK {
 public:
  static bool IKinBody(const Eigen::MatrixXd& Blist,
                       const Eigen::MatrixXd& M,
                       const Eigen::MatrixXd& T,
                       Eigen::VectorXd& thetalist,
                       double eomg,
                       double ev);
  static bool IKinSpace(const Eigen::MatrixXd& Slist,
                        const Eigen::MatrixXd& M,
                        const Eigen::MatrixXd& T,
                        Eigen::VectorXd& thetalist,
                        double eomg,
                        double ev);
};
}  // namespace mymr
