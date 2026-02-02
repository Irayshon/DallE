#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
class Tools {
 public:
  static bool NearZero(double value);
  static Eigen::MatrixXd Normalize(Eigen::MatrixXd V);
  static Eigen::MatrixXd RotInv(const Eigen::MatrixXd& rot_matrix);
  static Eigen::Matrix3d VecToso3(const Eigen::Vector3d& omg);
  static Eigen::Vector3d so3ToVec(const Eigen::MatrixXd& so3mat);
  static Eigen::Vector4d AxisAng3(const Eigen::Vector3d& expc3);
  static Eigen::Matrix3d MatrixExp3(const Eigen::Matrix3d& so3mat);
  static Eigen::Matrix3d MatrixLog3(const Eigen::Matrix3d& R);
  static Eigen::MatrixXd RpToTrans(const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& p);
  static std::vector<Eigen::MatrixXd> TransToRp(const Eigen::MatrixXd& T);
  static Eigen::MatrixXd TransInv(const Eigen::MatrixXd& transform);
  static Eigen::MatrixXd VecTose3(const Eigen::VectorXd& V);
  static Eigen::VectorXd se3ToVec(const Eigen::MatrixXd& T);
  static Eigen::MatrixXd Adjoint(const Eigen::MatrixXd& T);
  static Eigen::VectorXd ScrewToAxis(Eigen::Vector3d q, Eigen::Vector3d s,
                                     double h);
  static Eigen::VectorXd AxisAng6(const Eigen::VectorXd& expc6);
  static Eigen::MatrixXd MatrixExp6(const Eigen::MatrixXd& se3mat);
  static Eigen::MatrixXd MatrixLog6(const Eigen::MatrixXd& T);
  static Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d& R);
  static Eigen::MatrixXd ProjectToSE3(const Eigen::MatrixXd& T);
  static double DistanceToSO3(const Eigen::Matrix3d& R);
  static double DistanceToSE3(const Eigen::MatrixXd& T);
  static bool TestIfSO3(const Eigen::Matrix3d& R);
  static bool TestIfSE3(const Eigen::MatrixXd& T);
  static Eigen::MatrixXd JacobianSpace(const Eigen::MatrixXd& Slist,
                                       const Eigen::VectorXd& thetalist);
  static Eigen::MatrixXd JacobianBody(const Eigen::MatrixXd& Blist,
                                      const Eigen::VectorXd& thetalist);
  static Eigen::MatrixXd ad(const Eigen::VectorXd& V);
  static bool IsSingular(const Eigen::MatrixXd& J);
  static double ConditionNumber(const Eigen::MatrixXd& J);
};
}  // namespace mymr
