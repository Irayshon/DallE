#include <gtest/gtest.h>

#include <cstddef>

#include "robot_commander_cpp/quadruped_targets.hpp"

TEST(QuadrupedTargets, MatchesPythonCommanderData) {
  EXPECT_EQ(robot_commander_cpp::kQuadrupedJointNames.size(), 12u);
  EXPECT_EQ(robot_commander_cpp::kQuadrupedTargetPositions.size(), 12u);

  EXPECT_STREQ(robot_commander_cpp::kQuadrupedJointNames[0], "FR_hip_joint");
  EXPECT_STREQ(robot_commander_cpp::kQuadrupedJointNames[1], "FR_thigh_joint");
  EXPECT_STREQ(robot_commander_cpp::kQuadrupedJointNames[2], "FR_calf_joint");
  EXPECT_STREQ(robot_commander_cpp::kQuadrupedJointNames[9], "RL_hip_joint");
  EXPECT_STREQ(robot_commander_cpp::kQuadrupedJointNames[10], "RL_thigh_joint");
  EXPECT_STREQ(robot_commander_cpp::kQuadrupedJointNames[11], "RL_calf_joint");

  for (std::size_t i = 0; i < robot_commander_cpp::kQuadrupedTargetPositions.size(); ++i) {
    if (i % 3 == 2) {
      EXPECT_DOUBLE_EQ(robot_commander_cpp::kQuadrupedTargetPositions[i], -1.7);
    } else {
      EXPECT_DOUBLE_EQ(robot_commander_cpp::kQuadrupedTargetPositions[i], 0.0);
    }
  }

  EXPECT_EQ(robot_commander_cpp::kQuadrupedTargetSeconds, 3);
}
