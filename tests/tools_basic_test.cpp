#include "WallE/tools.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <type_traits>

static_assert(
    std::is_same_v<decltype(WallE::Tools::Normalize(Eigen::MatrixXd())),
                   Eigen::MatrixXd>,
    "Tools::Normalize should take and return Eigen::MatrixXd by value");

TEST(ToolsBasicTest, NearZeroAndNormalize) {
  EXPECT_TRUE(WallE::Tools::NearZero(1e-7));
  Eigen::MatrixXd v(3, 1);
  v << 1, 2, 2;
  auto n = WallE::Tools::Normalize(v);
  EXPECT_NEAR(n.norm(), 1.0, 1e-9);
}

TEST(ToolsBasicTest, ProjectToSo3Se3Validity) {
  Eigen::Vector3d w(1.0, 2.0, 3.0);
  w.normalize();
  Eigen::Matrix3d so3 = WallE::Tools::VecToso3(w * 0.6);
  Eigen::Matrix3d R = WallE::Tools::MatrixExp3(so3);
  Eigen::Matrix3d R_bad = R;
  R_bad(0, 0) += 0.02;
  R_bad(1, 2) -= 0.01;

  Eigen::Matrix3d R_proj = WallE::Tools::ProjectToSO3(R_bad);
  EXPECT_TRUE(WallE::Tools::TestIfSO3(R_proj));

  Eigen::Vector3d p(0.3, -0.2, 0.5);
  Eigen::MatrixXd T_bad = WallE::Tools::RpToTrans(R_bad, p);
  Eigen::MatrixXd T_proj = WallE::Tools::ProjectToSE3(T_bad);
  EXPECT_TRUE(WallE::Tools::TestIfSE3(T_proj));
}
