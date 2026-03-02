#include <gtest/gtest.h>

#include "robot_commander_cpp/arm7dof_targets.hpp"

TEST(Arm7dofTargets, MatchesPythonJointCommanderData) {
  EXPECT_EQ(robot_commander_cpp::kArm7dofJointNames.size(), 7u);
  EXPECT_EQ(robot_commander_cpp::kArm7dofTargetPositions.size(), 7u);

  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[0], "joint1");
  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[1], "joint3");
  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[2], "joint5");
  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[3], "joint6");
  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[4], "joint7");
  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[5], "joint8");
  EXPECT_STREQ(robot_commander_cpp::kArm7dofJointNames[6], "joint9");

  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[0], 3.5);
  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[1], -1.5);
  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[2], 0.5);
  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[3], 0.5);
  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[4], 0.5);
  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[5], 0.5);
  EXPECT_DOUBLE_EQ(robot_commander_cpp::kArm7dofTargetPositions[6], 0.5);
  EXPECT_EQ(robot_commander_cpp::kArm7dofTargetSeconds, 2);
}
