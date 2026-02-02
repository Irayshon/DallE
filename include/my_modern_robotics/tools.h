#pragma once

#include <Eigen/Dense>

namespace mymr {
class Tools {
 public:
  static bool NearZero(double value);
  static Eigen::MatrixXd Normalize(Eigen::MatrixXd V);
};
}  // namespace mymr
