#include "WallE/fk.h"
#include "WallE/ik.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

TEST(IKTest, BodyAndSpaceExample) {
  double pi = std::acos(-1.0);
  Eigen::Matrix4d M;
  M << -1, 0, 0, 0,
       0, 1, 0, 6,
       0, 0, -1, 2,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 3> Blist;
  Blist << 0, 0, 0,
           0, 0, 0,
           -1, 0, 1,
           2, 0, 0,
           0, 1, 0,
           0, 0, 0.1;

  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 0, 0, 0,
           0, 0, 0,
           1, 0, -1,
           4, 0, -6,
           0, 1, 0,
           0, 0, -0.1;

  Eigen::Matrix4d T;
  T << 0, 1, 0, -5,
       1, 0, 0, 4,
       0, 0, -1, 1.68584073,
       0, 0, 0, 1;

  Eigen::VectorXd thetalist0(3);
  thetalist0 << 1.5, 2.5, 3.0;

  Eigen::VectorXd thetalist_body = thetalist0;
  bool success_body = WallE::IK::IKinBody(Blist, M, T, thetalist_body, 1e-3, 1e-3);
  EXPECT_TRUE(success_body);
  auto T_body = WallE::FK::FKinBody(M, Blist, thetalist_body);

  Eigen::VectorXd thetalist_space = thetalist0;
  bool success_space = WallE::IK::IKinSpace(Slist, M, T, thetalist_space, 1e-3, 1e-3);
  EXPECT_TRUE(success_space);
  auto T_space = WallE::FK::FKinSpace(M, Slist, thetalist_space);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(T_body(r, c), T(r, c), 1e-3);
      EXPECT_NEAR(T_space(r, c), T(r, c), 1e-3);
    }
  }
}

TEST(IKTest, PlanarTwoLinkTarget) {
  double pi = std::acos(-1.0);
  Eigen::Matrix4d M;
  M << 1, 0, 0, 2,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 2> Slist;
  Slist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           0, -1,
           0, 0;

  Eigen::Matrix<double, 6, 2> Blist;
  Blist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           2, 1,
           0, 0;

  double theta1 = pi / 4.0;
  double theta2 = -pi / 3.0;
  double theta12 = theta1 + theta2;
  double c12 = std::cos(theta12);
  double s12 = std::sin(theta12);
  double x = std::cos(theta1) + std::cos(theta12);
  double y = std::sin(theta1) + std::sin(theta12);

  Eigen::Matrix4d T;
  T << c12, -s12, 0, x,
       s12, c12, 0, y,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::VectorXd thetalist0(2);
  thetalist0 << 0.7, -1.0;

  Eigen::VectorXd thetalist_body = thetalist0;
  bool success_body = WallE::IK::IKinBody(Blist, M, T, thetalist_body, 1e-4, 1e-4);
  EXPECT_TRUE(success_body);
  auto T_body = WallE::FK::FKinBody(M, Blist, thetalist_body);

  Eigen::VectorXd thetalist_space = thetalist0;
  bool success_space = WallE::IK::IKinSpace(Slist, M, T, thetalist_space, 1e-4, 1e-4);
  EXPECT_TRUE(success_space);
  auto T_space = WallE::FK::FKinSpace(M, Slist, thetalist_space);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(T_body(r, c), T(r, c), 1e-4);
      EXPECT_NEAR(T_space(r, c), T(r, c), 1e-4);
    }
  }
}

TEST(MPCIKTest, MPCPlanarTwoLinkTrajectory) {
  double pi = std::acos(-1.0);
  Eigen::Matrix4d M;
  M << 1, 0, 0, 2,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 2> Blist;
  Blist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           2, 1,
           0, 0;

  Eigen::Matrix<double, 6, 2> Slist;
  Slist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           0, -1,
           0, 0;

  Eigen::VectorXd theta_a(2);
  theta_a << pi / 6.0, -pi / 6.0;
  Eigen::VectorXd theta_b(2);
  theta_b << pi / 4.0, -pi / 4.0;
  Eigen::VectorXd theta_c(2);
  theta_c << pi / 3.0, -pi / 3.0;

  Eigen::Matrix4d T_a = WallE::FK::FKinBody(M, Blist, theta_a);
  Eigen::Matrix4d T_b = WallE::FK::FKinBody(M, Blist, theta_b);
  Eigen::Matrix4d T_c = WallE::FK::FKinBody(M, Blist, theta_c);
  Eigen::Matrix4d T_c_space = WallE::FK::FKinSpace(M, Slist, theta_c);
  EXPECT_NEAR(T_c_space(0, 3), T_c(0, 3), 1e-9);
  EXPECT_NEAR(T_c_space(1, 3), T_c(1, 3), 1e-9);

  std::vector<Eigen::Matrix4d> trajectory = {T_a, T_b, T_c};

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 200;

  Eigen::VectorXd thetalist(2);
  thetalist << 0.1, -0.1;
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, thetalist, config);

  EXPECT_TRUE(result.converged);
  Eigen::MatrixXd T_final = WallE::FK::FKinBody(M, Blist, result.thetalist);
  EXPECT_NEAR(T_final(0, 3), trajectory.back()(0, 3), 1e-2);
  EXPECT_NEAR(T_final(1, 3), trajectory.back()(1, 3), 1e-2);
  EXPECT_NEAR(T_final(2, 3), trajectory.back()(2, 3), 1e-2);
}

TEST(MPCIKTest, MPCThreeDOFJointLimits) {
  Eigen::Matrix4d M;
  M << -1, 0, 0, 0,
       0, 1, 0, 6,
       0, 0, -1, 2,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 3> Blist;
  Blist << 0, 0, 0,
           0, 0, 0,
           -1, 0, 1,
           2, 0, 0,
           0, 1, 0,
           0, 0, 0.1;

  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 0, 0, 0,
           0, 0, 0,
           1, 0, -1,
           4, 0, -6,
           0, 1, 0,
           0, 0, -0.1;

  Eigen::VectorXd theta_target(3);
  theta_target << 0.5, 0.5, 0.5;
  Eigen::Matrix4d T_target = WallE::FK::FKinBody(M, Blist, theta_target);
  Eigen::Matrix4d T_target_space = WallE::FK::FKinSpace(M, Slist, theta_target);
  EXPECT_NEAR(T_target_space(0, 3), T_target(0, 3), 1e-9);
  EXPECT_NEAR(T_target_space(1, 3), T_target(1, 3), 1e-9);
  EXPECT_NEAR(T_target_space(2, 3), T_target(2, 3), 1e-9);

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 300;
  config.eomg = 1e-2;
  config.ev = 1e-2;
  config.theta_min = Eigen::Vector3d(-1.0, -1.0, -1.0);
  config.theta_max = Eigen::Vector3d(1.0, 1.0, 1.0);
  config.dtheta_min = Eigen::Vector3d(-2.0, -2.0, -2.0);
  config.dtheta_max = Eigen::Vector3d(2.0, 2.0, 2.0);

  Eigen::VectorXd thetalist(3);
  thetalist << 0.0, 0.0, 0.0;
  std::vector<Eigen::Matrix4d> trajectory = {T_target};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, thetalist, config);

  EXPECT_EQ(result.thetalist.size(), 3);
  for (int i = 0; i < result.thetalist.size(); ++i) {
    EXPECT_GE(result.thetalist(i), config.theta_min(i) - 1e-6);
    EXPECT_LE(result.thetalist(i), config.theta_max(i) + 1e-6);
  }
}

TEST(MPCIKTest, MPCSingularityAvoidance) {
  Eigen::Matrix4d M;
  M << -1, 0, 0, 0,
       0, 1, 0, 6,
       0, 0, -1, 2,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 3> Blist;
  Blist << 0, 0, 0,
           0, 0, 0,
           -1, 0, 1,
           2, 0, 0,
           0, 1, 0,
           0, 0, 0.1;

  Eigen::VectorXd theta_target(3);
  theta_target << 0.3, 0.5, -0.3;
  Eigen::Matrix4d T_target = WallE::FK::FKinBody(M, Blist, theta_target);

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 300;
  config.eomg = 1e-2;
  config.ev = 1e-2;
  config.manipulability_threshold = 0.001;

  Eigen::VectorXd thetalist(3);
  thetalist << 0.0, 0.0, 0.0;
  std::vector<Eigen::Matrix4d> trajectory = {T_target};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, thetalist, config);

  EXPECT_GT(result.iterations, 0);
  EXPECT_EQ(result.thetalist.size(), 3);
  for (int i = 0; i < result.thetalist.size(); ++i) {
    EXPECT_TRUE(std::isfinite(result.thetalist(i)));
  }
  EXPECT_TRUE(std::isfinite(result.final_position_error));
  EXPECT_TRUE(std::isfinite(result.final_orientation_error));
}

TEST(MPCIKTest, MPCWorkspaceBounds) {
  double pi = std::acos(-1.0);
  Eigen::Matrix4d M;
  M << 1, 0, 0, 2,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 2> Blist;
  Blist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           2, 1,
           0, 0;

  Eigen::VectorXd theta_target(2);
  theta_target << pi / 4.0, -pi / 4.0;
  Eigen::Matrix4d T_target = WallE::FK::FKinBody(M, Blist, theta_target);

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 200;
  config.eomg = 1e-2;
  config.ev = 1e-2;
  config.workspace_min = Eigen::Vector3d(-0.5, -0.5, -1.0);
  config.workspace_max = Eigen::Vector3d(2.0, 2.0, 1.0);

  Eigen::VectorXd thetalist(2);
  thetalist << 0.1, -0.1;
  std::vector<Eigen::Matrix4d> trajectory = {T_target};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, thetalist, config);

  Eigen::MatrixXd T_final = WallE::FK::FKinBody(M, Blist, result.thetalist);
  Eigen::Vector3d p_final = T_final.block<3, 1>(0, 3);
  bool within_workspace =
      (p_final.array() >= config.workspace_min.array() - 1e-6).all() &&
      (p_final.array() <= config.workspace_max.array() + 1e-6).all();

  EXPECT_TRUE(result.converged || within_workspace);
  if (!result.converged) {
    EXPECT_TRUE(within_workspace);
  }
}

TEST(MPCIKTest, MPCSinglePoseConvergence) {
  Eigen::Matrix4d M;
  M << -1, 0, 0, 0,
       0, 1, 0, 6,
       0, 0, -1, 2,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 3> Blist;
  Blist << 0, 0, 0,
           0, 0, 0,
           -1, 0, 1,
           2, 0, 0,
           0, 1, 0,
           0, 0, 0.1;

  Eigen::Matrix4d T;
  T << 0, 1, 0, -5,
       1, 0, 0, 4,
       0, 0, -1, 1.68584073,
       0, 0, 0, 1;

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 300;

  Eigen::VectorXd thetalist(3);
  thetalist << 1.5, 2.5, 3.0;
  std::vector<Eigen::Matrix4d> trajectory = {T};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, thetalist, config);

  EXPECT_TRUE(result.converged);
  Eigen::MatrixXd T_final = WallE::FK::FKinBody(M, Blist, result.thetalist);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(T_final(r, c), T(r, c), 1e-2);
    }
  }
  EXPECT_NEAR(T_final(0, 3), T(0, 3), 1e-2);
  EXPECT_NEAR(T_final(1, 3), T(1, 3), 1e-2);
  EXPECT_NEAR(T_final(2, 3), T(2, 3), 1e-2);
}

TEST(MPCIKTest, MPCBodySpaceConsistency) {
  Eigen::Matrix4d M;
  M << -1, 0, 0, 0,
       0, 1, 0, 6,
       0, 0, -1, 2,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 3> Blist;
  Blist << 0, 0, 0,
           0, 0, 0,
           -1, 0, 1,
           2, 0, 0,
           0, 1, 0,
           0, 0, 0.1;

  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 0, 0, 0,
           0, 0, 0,
           1, 0, -1,
           4, 0, -6,
           0, 1, 0,
           0, 0, -0.1;

  Eigen::Matrix4d T;
  T << 0, 1, 0, -5,
       1, 0, 0, 4,
       0, 0, -1, 1.68584073,
       0, 0, 0, 1;

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 300;

  Eigen::VectorXd theta_body(3);
  theta_body << 1.5, 2.5, 3.0;
  Eigen::VectorXd theta_space = theta_body;
  std::vector<Eigen::Matrix4d> trajectory = {T};

  WallE::MPCIKResult result_body =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, theta_body, config);
  WallE::MPCIKResult result_space =
      WallE::IK::MPCIKinSpace(Slist, M, trajectory, theta_space, config);

  EXPECT_TRUE(result_body.converged);
  EXPECT_TRUE(result_space.converged);
  Eigen::MatrixXd T_body = WallE::FK::FKinBody(M, Blist, result_body.thetalist);
  Eigen::MatrixXd T_space = WallE::FK::FKinSpace(M, Slist, result_space.thetalist);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      EXPECT_NEAR(T_body(r, c), T_space(r, c), 1e-2);
    }
  }
}

TEST(MPCIKTest, MPCNonConvergence) {
  Eigen::Matrix4d M;
  M << 1, 0, 0, 2,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  Eigen::Matrix<double, 6, 2> Blist;
  Blist << 0, 0,
           0, 0,
           1, 1,
           0, 0,
           2, 1,
           0, 0;

  Eigen::Matrix4d T_far;
  T_far << 1, 0, 0, 10,
           0, 1, 0, 10,
           0, 0, 1, 0,
           0, 0, 0, 1;

  WallE::MPCIKConfig config;
  config.horizon = 5;
  config.dt = 0.05;
  config.max_iterations = 50;

  Eigen::VectorXd thetalist(2);
  thetalist << 0.0, 0.0;
  std::vector<Eigen::Matrix4d> trajectory = {T_far};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinBody(Blist, M, trajectory, thetalist, config);

  EXPECT_FALSE(result.converged);
  EXPECT_GT(result.final_position_error, 1.0);
  EXPECT_TRUE(std::isfinite(result.final_position_error));
}

TEST(MPCIKTest, MPCSevenDOFArm) {
  Eigen::Matrix<double, 6, 7> Slist;
  Slist << 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, -1, 0, -1, 0,
           1, 0, 1, 0, 1, 0, -1,
           0, -0.333, 0, 0.649, 0, 1.033, 0,
           0, 0, 0, 0, 0, 0, 0.088,
           0, 0, 0, -0.0825, 0, 0, 0;

  Eigen::Matrix4d M;
  M << 1, 0, 0, 0.088,
       0, 1, 0, 0,
       0, 0, 1, 1.033,
       0, 0, 0, 1;

  Eigen::VectorXd theta_target(7);
  theta_target << 0.1, -0.2, 0.15, -0.3, 0.1, 0.2, -0.1;
  Eigen::Matrix4d T_target = WallE::FK::FKinSpace(M, Slist, theta_target);

  WallE::MPCIKConfig config;
  config.horizon = 3;
  config.dt = 0.05;
  config.max_iterations = 300;
  config.eomg = 5e-2;
  config.ev = 5e-2;
  config.theta_min = Eigen::VectorXd::Constant(7, -2.8);
  config.theta_max = Eigen::VectorXd::Constant(7, 2.8);
  config.dtheta_min = Eigen::VectorXd::Constant(7, -2.0);
  config.dtheta_max = Eigen::VectorXd::Constant(7, 2.0);

  Eigen::VectorXd thetalist = Eigen::VectorXd::Zero(7);
  std::vector<Eigen::Matrix4d> trajectory = {T_target};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinSpace(Slist, M, trajectory, thetalist, config);

  EXPECT_GT(result.iterations, 0);
  EXPECT_EQ(result.thetalist.size(), 7);
  for (int i = 0; i < result.thetalist.size(); ++i) {
    EXPECT_TRUE(std::isfinite(result.thetalist(i)));
  }

  if (result.converged) {
    for (int i = 0; i < result.thetalist.size(); ++i) {
      EXPECT_GE(result.thetalist(i), config.theta_min(i) - 1e-6);
      EXPECT_LE(result.thetalist(i), config.theta_max(i) + 1e-6);
    }
  }
}

TEST(MPCIKTest, MPCQuadrupedLeg) {
  Eigen::Matrix<double, 6, 3> Slist;
  Slist << 1, 0, 0,
           0, 1, 1,
           0, 0, 0,
           0, 0, 0.2,
           0, 0, 0,
           0, -0.08, -0.08;

  Eigen::Matrix4d M;
  M << 1, 0, 0, -0.08,
       0, 1, 0, 0,
       0, 0, 1, -0.213,
       0, 0, 0, 1;

  Eigen::VectorXd theta_a(3);
  theta_a << 0.0, 0.3, -0.6;
  Eigen::VectorXd theta_b(3);
  theta_b << 0.0, 0.5, -1.0;
  Eigen::VectorXd theta_c(3);
  theta_c << 0.0, 0.2, -0.4;
  Eigen::Matrix4d T_a = WallE::FK::FKinSpace(M, Slist, theta_a);
  Eigen::Matrix4d T_b = WallE::FK::FKinSpace(M, Slist, theta_b);
  Eigen::Matrix4d T_c = WallE::FK::FKinSpace(M, Slist, theta_c);

  WallE::MPCIKConfig config;
  config.horizon = 3;
  config.dt = 0.02;
  config.max_iterations = 300;
  config.eomg = 5e-2;
  config.ev = 5e-2;
  config.theta_min = Eigen::Vector3d(-0.8, -1.0, -2.7);
  config.theta_max = Eigen::Vector3d(0.8, 3.5, -0.9);
  config.dtheta_min = Eigen::Vector3d(-10.0, -10.0, -10.0);
  config.dtheta_max = Eigen::Vector3d(10.0, 10.0, 10.0);

  Eigen::VectorXd thetalist(3);
  thetalist << 0.0, 0.0, 0.0;
  std::vector<Eigen::Matrix4d> trajectory = {T_a, T_b, T_c};
  WallE::MPCIKResult result =
      WallE::IK::MPCIKinSpace(Slist, M, trajectory, thetalist, config);

  EXPECT_GT(result.iterations, 0);
  EXPECT_EQ(result.thetalist.size(), 3);
  for (int i = 0; i < result.thetalist.size(); ++i) {
    EXPECT_TRUE(std::isfinite(result.thetalist(i)));
  }

  if (result.converged) {
    for (int i = 0; i < result.thetalist.size(); ++i) {
      EXPECT_GE(result.thetalist(i), config.theta_min(i) - 1e-6);
      EXPECT_LE(result.thetalist(i), config.theta_max(i) + 1e-6);
    }
  }
}
