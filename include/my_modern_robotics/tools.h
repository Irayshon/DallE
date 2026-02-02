#pragma once

#include <Eigen/Dense>

namespace mymr {
class Tools {
 public:
  static bool NearZero(double value);
  static Eigen::VectorXd Normalize(const Eigen::VectorXd& vector);
};
}  // namespace mymr
