#include "WallE/tools.h"

#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

namespace WallE {
bool Tools::NearZero(double value) {
  return std::abs(value) < 1e-6;
}

Eigen::MatrixXd Tools::Normalize(Eigen::MatrixXd V) {
  return V / V.norm();
}

Eigen::MatrixXd Tools::RotInv(const Eigen::MatrixXd& rot_matrix) {
  return rot_matrix.transpose();
}

Eigen::Matrix3d Tools::VecToso3(const Eigen::Vector3d& omg) {
  Eigen::Matrix3d m_ret;
  m_ret << 0, -omg(2), omg(1),
      omg(2), 0, -omg(0),
      -omg(1), omg(0), 0;
  return m_ret;
}

Eigen::Vector3d Tools::so3ToVec(const Eigen::MatrixXd& so3mat) {
  Eigen::Vector3d v_ret;
  v_ret << so3mat(2, 1), so3mat(0, 2), so3mat(1, 0);
  return v_ret;
}

Eigen::Vector4d Tools::AxisAng3(const Eigen::Vector3d& expc3) {
  Eigen::Vector4d v_ret;
  v_ret << expc3.normalized(), expc3.norm();
  return v_ret;
}

Eigen::Matrix3d Tools::MatrixExp3(const Eigen::Matrix3d& so3mat) {
  Eigen::Vector3d omgtheta = so3ToVec(so3mat);
  Eigen::Matrix3d m_ret = Eigen::Matrix3d::Identity();
  if (NearZero(so3mat.norm())) {
    return m_ret;
  }

  double theta = (AxisAng3(omgtheta))(3);
  Eigen::Matrix3d omgmat = so3mat * (1 / theta);
  return m_ret + std::sin(theta) * omgmat +
         ((1 - std::cos(theta)) * (omgmat * omgmat));
}

Eigen::Matrix3d Tools::MatrixLog3(const Eigen::Matrix3d& R) {
  double acosinput = (R.trace() - 1) / 2.0;
  Eigen::Matrix3d m_ret = Eigen::Matrix3d::Zero();
  if (acosinput >= 1) {
    return m_ret;
  }
  if (acosinput <= -1) {
    Eigen::Vector3d omg;
    if (!NearZero(1 + R(2, 2))) {
      omg = (1.0 / std::sqrt(2 * (1 + R(2, 2)))) *
            Eigen::Vector3d(R(0, 2), R(1, 2), 1 + R(2, 2));
    } else if (!NearZero(1 + R(1, 1))) {
      omg = (1.0 / std::sqrt(2 * (1 + R(1, 1)))) *
            Eigen::Vector3d(R(0, 1), 1 + R(1, 1), R(2, 1));
    } else {
      omg = (1.0 / std::sqrt(2 * (1 + R(0, 0)))) *
            Eigen::Vector3d(1 + R(0, 0), R(1, 0), R(2, 0));
    }
    m_ret = VecToso3(kPi * omg);
    return m_ret;
  }

  double theta = std::acos(acosinput);
  m_ret = theta / 2.0 / std::sin(theta) * (R - R.transpose());
  return m_ret;
}

Eigen::MatrixXd Tools::RpToTrans(const Eigen::Matrix3d& R,
                                 const Eigen::Vector3d& p) {
  Eigen::MatrixXd m_ret(4, 4);
  m_ret << R, p,
      0, 0, 0, 1;
  return m_ret;
}

std::vector<Eigen::MatrixXd> Tools::TransToRp(const Eigen::MatrixXd& T) {
  std::vector<Eigen::MatrixXd> Rp_ret;
  Eigen::Matrix3d R_ret = T.block<3, 3>(0, 0);
  Eigen::Vector3d p_ret(T(0, 3), T(1, 3), T(2, 3));
  Rp_ret.push_back(R_ret);
  Rp_ret.push_back(p_ret);
  return Rp_ret;
}

Eigen::MatrixXd Tools::TransInv(const Eigen::MatrixXd& transform) {
  auto rp = Tools::TransToRp(transform);
  Eigen::Matrix3d Rt = rp.at(0).transpose();
  Eigen::Vector3d t = -(Rt * rp.at(1));
  Eigen::MatrixXd inv = Eigen::MatrixXd::Zero(4, 4);
  inv.block(0, 0, 3, 3) = Rt;
  inv.block(0, 3, 3, 1) = t;
  inv(3, 3) = 1;
  return inv;
}

Eigen::MatrixXd Tools::VecTose3(const Eigen::VectorXd& V) {
  Eigen::Vector3d exp(V(0), V(1), V(2));
  Eigen::Vector3d linear(V(3), V(4), V(5));
  Eigen::MatrixXd m_ret(4, 4);
  m_ret << VecToso3(exp), linear,
      0, 0, 0, 0;
  return m_ret;
}

Eigen::VectorXd Tools::se3ToVec(const Eigen::MatrixXd& T) {
  Eigen::VectorXd m_ret(6);
  m_ret << T(2, 1), T(0, 2), T(1, 0), T(0, 3), T(1, 3), T(2, 3);
  return m_ret;
}

Eigen::MatrixXd Tools::Adjoint(const Eigen::MatrixXd& T) {
  auto rp = Tools::TransToRp(T);
  Eigen::MatrixXd ad_ret = Eigen::MatrixXd::Zero(6, 6);
  Eigen::MatrixXd zeroes = Eigen::MatrixXd::Zero(3, 3);
  ad_ret << rp[0], zeroes,
      VecToso3(rp[1]) * rp[0], rp[0];
  return ad_ret;
}

Eigen::VectorXd Tools::ScrewToAxis(Eigen::Vector3d q, Eigen::Vector3d s,
                                   double h) {
  Eigen::VectorXd axis(6);
  axis.segment(0, 3) = s;
  axis.segment(3, 3) = q.cross(s) + (h * s);
  return axis;
}

Eigen::VectorXd Tools::AxisAng6(const Eigen::VectorXd& expc6) {
  Eigen::VectorXd v_ret(7);
  double theta = Eigen::Vector3d(expc6(0), expc6(1), expc6(2)).norm();
  if (NearZero(theta)) {
    theta = Eigen::Vector3d(expc6(3), expc6(4), expc6(5)).norm();
  }
  v_ret << expc6 / theta, theta;
  return v_ret;
}

Eigen::MatrixXd Tools::MatrixExp6(const Eigen::MatrixXd& se3mat) {
  Eigen::Matrix3d se3mat_cut = se3mat.block<3, 3>(0, 0);
  Eigen::Vector3d omgtheta = so3ToVec(se3mat_cut);
  Eigen::MatrixXd m_ret(4, 4);
  if (NearZero(omgtheta.norm())) {
    se3mat_cut = Eigen::MatrixXd::Identity(3, 3);
    omgtheta << se3mat(0, 3), se3mat(1, 3), se3mat(2, 3);
    m_ret << se3mat_cut, omgtheta,
        0, 0, 0, 1;
    return m_ret;
  }

  double theta = (AxisAng3(omgtheta))(3);
  Eigen::Matrix3d omgmat = se3mat.block<3, 3>(0, 0) / theta;
  Eigen::Matrix3d expExpand = Eigen::MatrixXd::Identity(3, 3) * theta +
                              (1 - std::cos(theta)) * omgmat +
                              ((theta - std::sin(theta)) * (omgmat * omgmat));
  Eigen::Vector3d linear(se3mat(0, 3), se3mat(1, 3), se3mat(2, 3));
  Eigen::Vector3d GThetaV = (expExpand * linear) / theta;
  m_ret << MatrixExp3(se3mat_cut), GThetaV,
      0, 0, 0, 1;
  return m_ret;
}

Eigen::MatrixXd Tools::MatrixLog6(const Eigen::MatrixXd& T) {
  Eigen::MatrixXd m_ret(4, 4);
  auto rp = Tools::TransToRp(T);
  Eigen::Matrix3d omgmat = MatrixLog3(rp.at(0));
  Eigen::Matrix3d zeros3d = Eigen::Matrix3d::Zero();
  if (NearZero(omgmat.norm())) {
    m_ret << zeros3d, rp.at(1),
        0, 0, 0, 0;
  } else {
    double theta = std::acos((rp.at(0).trace() - 1) / 2.0);
    Eigen::Matrix3d logExpand1 = Eigen::MatrixXd::Identity(3, 3) - omgmat / 2.0;
    Eigen::Matrix3d logExpand2 =
        (1.0 / theta - 1.0 / std::tan(theta / 2.0) / 2) * omgmat * omgmat /
        theta;
    Eigen::Matrix3d logExpand = logExpand1 + logExpand2;
    m_ret << omgmat, logExpand * rp.at(1),
        0, 0, 0, 0;
  }
  return m_ret;
}

Eigen::Matrix3d Tools::ProjectToSO3(const Eigen::Matrix3d& R) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R,
                                       Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d R_proj = U * V.transpose();
  if (R_proj.determinant() < 0) {
    U.col(2) *= -1.0;
    R_proj = U * V.transpose();
  }
  return R_proj;
}

Eigen::MatrixXd Tools::ProjectToSE3(const Eigen::MatrixXd& T) {
  Eigen::Matrix3d R = ProjectToSO3(T.block<3, 3>(0, 0));
  Eigen::Vector3d p = T.block<3, 1>(0, 3);
  return RpToTrans(R, p);
}

double Tools::DistanceToSO3(const Eigen::Matrix3d& R) {
  return (R - ProjectToSO3(R)).norm();
}

double Tools::DistanceToSE3(const Eigen::MatrixXd& T) {
  return (T - ProjectToSE3(T)).norm();
}

bool Tools::TestIfSO3(const Eigen::Matrix3d& R) {
  return DistanceToSO3(R) < 1e-3;
}

bool Tools::TestIfSE3(const Eigen::MatrixXd& T) {
  return DistanceToSE3(T) < 1e-3;
}

Eigen::MatrixXd Tools::JacobianSpace(const Eigen::MatrixXd& Slist,
                                     const Eigen::VectorXd& thetalist) {
  int n = static_cast<int>(thetalist.size());
  if (n == 0) {
    return Eigen::MatrixXd(6, 0);
  }

  Eigen::MatrixXd Js(6, n);
  Js.col(0) = Slist.col(0);
  Eigen::MatrixXd T = Eigen::MatrixXd::Identity(4, 4);
  for (int i = 1; i < n; ++i) {
    T = T * MatrixExp6(VecTose3(Slist.col(i - 1) * thetalist(i - 1)));
    Js.col(i) = Adjoint(T) * Slist.col(i);
  }
  return Js;
}

Eigen::MatrixXd Tools::JacobianBody(const Eigen::MatrixXd& Blist,
                                    const Eigen::VectorXd& thetalist) {
  int n = static_cast<int>(thetalist.size());
  if (n == 0) {
    return Eigen::MatrixXd(6, 0);
  }

  Eigen::MatrixXd Jb(6, n);
  Jb.col(n - 1) = Blist.col(n - 1);
  Eigen::MatrixXd T = Eigen::MatrixXd::Identity(4, 4);
  for (int i = n - 2; i >= 0; --i) {
    T = T * MatrixExp6(VecTose3(-Blist.col(i + 1) * thetalist(i + 1)));
    Jb.col(i) = Adjoint(T) * Blist.col(i);
  }
  return Jb;
}

Eigen::MatrixXd Tools::ad(const Eigen::VectorXd& V) {
  Eigen::Vector3d omg(V(0), V(1), V(2));
  Eigen::Vector3d v(V(3), V(4), V(5));
  Eigen::MatrixXd ad_ret = Eigen::MatrixXd::Zero(6, 6);
  ad_ret.block<3, 3>(0, 0) = VecToso3(omg);
  ad_ret.block<3, 3>(3, 0) = VecToso3(v);
  ad_ret.block<3, 3>(3, 3) = VecToso3(omg);
  return ad_ret;
}

bool Tools::IsSingular(const Eigen::MatrixXd& J) {
  return ConditionNumber(J) > 1e6;
}

double Tools::ConditionNumber(const Eigen::MatrixXd& J) {
  if (J.size() == 0) {
    return 0.0;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
  auto s = svd.singularValues();
  double min_sigma = s(s.size() - 1);
  if (NearZero(min_sigma)) {
    return std::numeric_limits<double>::infinity();
  }
  return s(0) / min_sigma;
}
}  // namespace WallE
