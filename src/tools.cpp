#include "my_modern_robotics/tools.h"

#include <cmath>

namespace mymr {
bool Tools::NearZero(double value) {
  return std::abs(value) < 1e-6;
}

Eigen::MatrixXd Tools::Normalize(Eigen::MatrixXd V) {
  return V / V.norm();
}
}  // namespace mymr
