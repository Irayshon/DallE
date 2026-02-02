#include "my_modern_robotics/tools.h"

#include <cmath>

namespace mymr {
bool Tools::NearZero(double value) {
  return std::abs(value) < 1e-6;
}

Eigen::VectorXd Tools::Normalize(const Eigen::VectorXd& vector) {
  return vector / vector.norm();
}
}  // namespace mymr
