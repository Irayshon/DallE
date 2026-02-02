#include "my_modern_robotics/ik.h"

#include <algorithm>

#include "my_modern_robotics/fk.h"
#include "my_modern_robotics/tools.h"

namespace {
Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& J) {
  if (J.size() == 0) {
    return Eigen::MatrixXd::Zero(J.cols(), J.rows());
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singular_values = svd.singularValues();
  double tolerance = 1e-6 * std::max(J.rows(), J.cols()) *
                     singular_values.array().abs()(0);
  Eigen::VectorXd singular_inverse = singular_values.unaryExpr(
      [tolerance](double sigma) { return sigma > tolerance ? 1.0 / sigma : 0.0; });
  return svd.matrixV() * singular_inverse.asDiagonal() *
         svd.matrixU().transpose();
}
}  // namespace

namespace mymr {
bool IK::IKinBody(const Eigen::MatrixXd& Blist,
                  const Eigen::MatrixXd& M,
                  const Eigen::MatrixXd& T,
                  Eigen::VectorXd& thetalist,
                  double eomg,
                  double ev) {
  const int max_iterations = 20;
  int iteration = 0;
  Eigen::MatrixXd Tsb = FK::FKinBody(M, Blist, thetalist);
  Eigen::VectorXd Vb = Tools::se3ToVec(
      Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
  bool err = (Vb.head(3).norm() > eomg) || (Vb.tail(3).norm() > ev);

  while (err && iteration < max_iterations) {
    Eigen::MatrixXd Jb = Tools::JacobianBody(Blist, thetalist);
    thetalist = thetalist + PseudoInverse(Jb) * Vb;
    ++iteration;
    Tsb = FK::FKinBody(M, Blist, thetalist);
    Vb = Tools::se3ToVec(Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
    err = (Vb.head(3).norm() > eomg) || (Vb.tail(3).norm() > ev);
  }

  return !err;
}

bool IK::IKinSpace(const Eigen::MatrixXd& Slist,
                   const Eigen::MatrixXd& M,
                   const Eigen::MatrixXd& T,
                   Eigen::VectorXd& thetalist,
                   double eomg,
                   double ev) {
  const int max_iterations = 20;
  int iteration = 0;
  Eigen::MatrixXd Tsb = FK::FKinSpace(M, Slist, thetalist);
  Eigen::VectorXd Vb = Tools::se3ToVec(
      Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
  Eigen::VectorXd Vs = Tools::Adjoint(Tsb) * Vb;
  bool err = (Vs.head(3).norm() > eomg) || (Vs.tail(3).norm() > ev);

  while (err && iteration < max_iterations) {
    Eigen::MatrixXd Js = Tools::JacobianSpace(Slist, thetalist);
    thetalist = thetalist + PseudoInverse(Js) * Vs;
    ++iteration;
    Tsb = FK::FKinSpace(M, Slist, thetalist);
    Vb = Tools::se3ToVec(Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
    Vs = Tools::Adjoint(Tsb) * Vb;
    err = (Vs.head(3).norm() > eomg) || (Vs.tail(3).norm() > ev);
  }

  return !err;
}
}  // namespace mymr
