#pragma once

#include <array>

namespace robot_commander_cpp {

inline constexpr std::array<const char *, 7> kArm7dofJointNames = {
  "joint1", "joint3", "joint5", "joint6", "joint7", "joint8", "joint9"
};

inline constexpr std::array<double, 7> kArm7dofTargetPositions = {
  3.5, -1.5, 0.5, 0.5, 0.5, 0.5, 0.5
};

inline constexpr int kArm7dofTargetSeconds = 2;

}
