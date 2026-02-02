#include "my_modern_robotics/tools.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <type_traits>

static_assert(
    std::is_same_v<decltype(mymr::Tools::Normalize(Eigen::MatrixXd())),
                   Eigen::MatrixXd>,
    "Tools::Normalize should take and return Eigen::MatrixXd by value");

TEST(ToolsBasicTest, NearZeroAndNormalize) {
  EXPECT_TRUE(mymr::Tools::NearZero(1e-7));
  Eigen::MatrixXd v(3, 1);
  v << 1, 2, 2;
  auto n = mymr::Tools::Normalize(v);
  EXPECT_NEAR(n.norm(), 1.0, 1e-9);
}

TEST(ToolsBasicTest, ProjectToSo3Se3Validity) {
  Eigen::Vector3d w(1.0, 2.0, 3.0);
  w.normalize();
  Eigen::Matrix3d so3 = mymr::Tools::VecToso3(w * 0.6);
  Eigen::Matrix3d R = mymr::Tools::MatrixExp3(so3);
  Eigen::Matrix3d R_bad = R;
  R_bad(0, 0) += 0.02;
  R_bad(1, 2) -= 0.01;

  Eigen::Matrix3d R_proj = mymr::Tools::ProjectToSO3(R_bad);
  EXPECT_TRUE(mymr::Tools::TestIfSO3(R_proj));

  Eigen::Vector3d p(0.3, -0.2, 0.5);
  Eigen::MatrixXd T_bad = mymr::Tools::RpToTrans(R_bad, p);
  Eigen::MatrixXd T_proj = mymr::Tools::ProjectToSE3(T_bad);
  EXPECT_TRUE(mymr::Tools::TestIfSE3(T_proj));
}
