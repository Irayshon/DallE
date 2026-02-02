#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
/**
 * @brief Mathematical helpers for rigid-body kinematics and dynamics.
 *
 * This class groups SE(3)/so(3) operations, screw theory helpers,
 * Jacobians, and validation utilities used across the library.
 */
class Tools {
 public:
  /**
   * @brief Check whether a scalar is close to zero.
   * @param value Scalar value to test.
   * @return True if the magnitude is below the internal tolerance.
   */
  static bool NearZero(double value);

  /**
   * @brief Normalize a vector or matrix by its Frobenius norm.
   * @param V Input matrix or vector.
   * @return Normalized matrix/vector.
   */
  static Eigen::MatrixXd Normalize(Eigen::MatrixXd V);

  /**
   * @brief Invert a rotation matrix (transpose).
   * @param rot_matrix Rotation matrix.
   * @return Inverse rotation matrix.
   */
  static Eigen::MatrixXd RotInv(const Eigen::MatrixXd& rot_matrix);

  /**
   * @brief Convert a 3-vector to a 3x3 so(3) matrix.
   * @param omg Angular velocity vector.
   * @return so(3) matrix.
   */
  static Eigen::Matrix3d VecToso3(const Eigen::Vector3d& omg);

  /**
   * @brief Convert a 3x3 so(3) matrix to a 3-vector.
   * @param so3mat so(3) matrix.
   * @return Angular velocity vector.
   */
  static Eigen::Vector3d so3ToVec(const Eigen::MatrixXd& so3mat);

  /**
   * @brief Convert exponential coordinates to axis-angle form in so(3).
   * @param expc3 Exponential coordinates (3-vector).
   * @return 4-vector [axis(3), theta].
   */
  static Eigen::Vector4d AxisAng3(const Eigen::Vector3d& expc3);

  /**
   * @brief Compute the matrix exponential of an so(3) matrix.
   * @param so3mat so(3) matrix.
   * @return Rotation matrix in SO(3).
   */
  static Eigen::Matrix3d MatrixExp3(const Eigen::Matrix3d& so3mat);

  /**
   * @brief Compute the matrix logarithm of an SO(3) rotation matrix.
   * @param R Rotation matrix in SO(3).
   * @return so(3) matrix.
   */
  static Eigen::Matrix3d MatrixLog3(const Eigen::Matrix3d& R);

  /**
   * @brief Form a homogeneous transform from rotation and position.
   * @param R Rotation matrix.
   * @param p Position vector.
   * @return 4x4 homogeneous transform.
   */
  static Eigen::MatrixXd RpToTrans(const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& p);

  /**
   * @brief Split a homogeneous transform into rotation and position.
   * @param T 4x4 homogeneous transform.
   * @return Vector of two matrices: rotation (3x3) and position (3x1).
   */
  static std::vector<Eigen::MatrixXd> TransToRp(const Eigen::MatrixXd& T);

  /**
   * @brief Invert a homogeneous transform.
   * @param transform 4x4 homogeneous transform.
   * @return Inverse transform.
   */
  static Eigen::MatrixXd TransInv(const Eigen::MatrixXd& transform);

  /**
   * @brief Convert a 6-vector to an se(3) matrix.
   * @param V 6-vector (twist).
   * @return 4x4 se(3) matrix.
   */
  static Eigen::MatrixXd VecTose3(const Eigen::VectorXd& V);

  /**
   * @brief Convert an se(3) matrix to a 6-vector.
   * @param T 4x4 se(3) matrix.
   * @return 6-vector twist.
   */
  static Eigen::VectorXd se3ToVec(const Eigen::MatrixXd& T);

  /**
   * @brief Compute the adjoint representation of a transform.
   * @param T 4x4 homogeneous transform.
   * @return 6x6 adjoint matrix.
   */
  static Eigen::MatrixXd Adjoint(const Eigen::MatrixXd& T);

  /**
   * @brief Construct a screw axis from a point, direction, and pitch.
   * @param q Point on the screw axis.
   * @param s Unit direction of the screw axis.
   * @param h Screw pitch.
   * @return 6-vector screw axis.
   */
  static Eigen::VectorXd ScrewToAxis(Eigen::Vector3d q, Eigen::Vector3d s,
                                     double h);

  /**
   * @brief Convert exponential coordinates in se(3) to axis-angle form.
   * @param expc6 Exponential coordinates (6-vector).
   * @return 7-vector [axis(6), theta].
   */
  static Eigen::VectorXd AxisAng6(const Eigen::VectorXd& expc6);

  /**
   * @brief Compute the matrix exponential of an se(3) matrix.
   * @param se3mat 4x4 se(3) matrix.
   * @return 4x4 homogeneous transform in SE(3).
   */
  static Eigen::MatrixXd MatrixExp6(const Eigen::MatrixXd& se3mat);

  /**
   * @brief Compute the matrix logarithm of an SE(3) transform.
   * @param T 4x4 homogeneous transform.
   * @return 4x4 se(3) matrix.
   */
  static Eigen::MatrixXd MatrixLog6(const Eigen::MatrixXd& T);

  /**
   * @brief Project a matrix to the closest SO(3) rotation.
   * @param R Input 3x3 matrix.
   * @return Rotation matrix in SO(3).
   */
  static Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d& R);

  /**
   * @brief Project a matrix to the closest SE(3) transform.
   * @param T Input 4x4 matrix.
   * @return Homogeneous transform in SE(3).
   */
  static Eigen::MatrixXd ProjectToSE3(const Eigen::MatrixXd& T);

  /**
   * @brief Distance from a matrix to SO(3).
   * @param R Input 3x3 matrix.
   * @return Scalar distance measure.
   */
  static double DistanceToSO3(const Eigen::Matrix3d& R);

  /**
   * @brief Distance from a matrix to SE(3).
   * @param T Input 4x4 matrix.
   * @return Scalar distance measure.
   */
  static double DistanceToSE3(const Eigen::MatrixXd& T);

  /**
   * @brief Check if a matrix is a valid SO(3) rotation.
   * @param R Input 3x3 matrix.
   * @return True if R is in SO(3).
   */
  static bool TestIfSO3(const Eigen::Matrix3d& R);

  /**
   * @brief Check if a matrix is a valid SE(3) transform.
   * @param T Input 4x4 matrix.
   * @return True if T is in SE(3).
   */
  static bool TestIfSE3(const Eigen::MatrixXd& T);

  /**
   * @brief Compute the space Jacobian.
   * @param Slist 6xn matrix of screw axes in space frame.
   * @param thetalist n-vector of joint angles.
   * @return 6xn Jacobian in space frame.
   */
  static Eigen::MatrixXd JacobianSpace(const Eigen::MatrixXd& Slist,
                                       const Eigen::VectorXd& thetalist);

  /**
   * @brief Compute the body Jacobian.
   * @param Blist 6xn matrix of screw axes in body frame.
   * @param thetalist n-vector of joint angles.
   * @return 6xn Jacobian in body frame.
   */
  static Eigen::MatrixXd JacobianBody(const Eigen::MatrixXd& Blist,
                                      const Eigen::VectorXd& thetalist);

  /**
   * @brief Compute the adjoint representation of a twist (ad operator).
   * @param V 6-vector twist.
   * @return 6x6 matrix ad(V).
   */
  static Eigen::MatrixXd ad(const Eigen::VectorXd& V);

  /**
   * @brief Check if a Jacobian is near-singular.
   * @param J Jacobian matrix.
   * @return True if the condition suggests singularity.
   */
  static bool IsSingular(const Eigen::MatrixXd& J);

  /**
   * @brief Compute the Jacobian condition number.
   * @param J Jacobian matrix.
   * @return Condition number (ratio of max/min singular value).
   */
  static double ConditionNumber(const Eigen::MatrixXd& J);
};
}  // namespace mymr
