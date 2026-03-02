#pragma once

#include <array>

namespace robot_commander_cpp {

inline constexpr std::array<const char *, 12> kQuadrupedJointNames = {
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
};

inline constexpr std::array<double, 12> kQuadrupedTargetPositions = {
  0.0, 0.0, -1.7,
  0.0, 0.0, -1.7,
  0.0, 0.0, -1.7,
  0.0, 0.0, -1.7
};

inline constexpr int kQuadrupedTargetSeconds = 3;

}
