#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
/**
 * @brief Forward dynamics and force-related helpers.
 */
class Dynamics {
 public:
  /**
   * @brief Compute the joint-space mass matrix.
   * @param thetalist n-vector of joint angles.
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return n x n mass matrix.
   */
  static Eigen::MatrixXd MassMatrix(
      const Eigen::VectorXd& thetalist,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute velocity quadratic forces (Coriolis/centripetal).
   * @param thetalist n-vector of joint angles.
   * @param dthetalist n-vector of joint velocities.
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return n-vector of joint forces/torques.
   */
  static Eigen::VectorXd VelQuadraticForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute gravity forces at the joints.
   * @param thetalist n-vector of joint angles.
   * @param g Gravity vector.
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return n-vector of joint forces/torques.
   */
  static Eigen::VectorXd GravityForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::Vector3d& g,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute joint forces from an end-effector wrench.
   * @param thetalist n-vector of joint angles.
   * @param Ftip Spatial force applied at the end-effector.
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return n-vector of joint forces/torques.
   */
  static Eigen::VectorXd EndEffectorForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute forward dynamics (joint accelerations).
   * @param thetalist n-vector of joint angles.
   * @param dthetalist n-vector of joint velocities.
   * @param taulist n-vector of joint forces/torques.
   * @param g Gravity vector.
   * @param Ftip Spatial force applied at the end-effector.
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return n-vector of joint accelerations.
   */
  static Eigen::VectorXd ForwardDynamics(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& taulist,
      const Eigen::Vector3d& g,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Integrate one Euler step for joint positions and velocities.
   * @param thetalist n-vector of joint angles.
   * @param dthetalist n-vector of joint velocities.
   * @param ddthetalist n-vector of joint accelerations.
   * @param dt Time step.
   * @return Vector containing updated thetalist and dthetalist.
   */
  static std::vector<Eigen::VectorXd> EulerStep(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& ddthetalist,
      double dt);

  /**
   * @brief Compute inverse dynamics for a joint-space trajectory.
   * @param thetamat Joint positions over time (rows).
   * @param dthetamat Joint velocities over time (rows).
   * @param ddthetamat Joint accelerations over time (rows).
   * @param g Gravity vector.
   * @param Ftipmat End-effector wrenches over time (rows).
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @return Matrix of joint torques over time (rows).
   */
  static Eigen::MatrixXd InverseDynamicsTrajectory(
      const Eigen::MatrixXd& thetamat,
      const Eigen::MatrixXd& dthetamat,
      const Eigen::MatrixXd& ddthetamat,
      const Eigen::Vector3d& g,
      const Eigen::MatrixXd& Ftipmat,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Simulate forward dynamics over a joint torque trajectory.
   * @param thetalist Initial joint angles.
   * @param dthetalist Initial joint velocities.
   * @param taumat Joint torque trajectory (rows).
   * @param g Gravity vector.
   * @param Ftipmat End-effector wrench trajectory (rows).
   * @param Mlist List of link frames {i} relative to {i-1} at home.
   * @param Glist Spatial inertia matrices of the links.
   * @param Slist 6xn screw axes in the space frame.
   * @param dt Time step between rows of taumat.
   * @param intRes Integration resolution per time step.
   * @return Vector of two matrices: joint positions and velocities (rows).
   */
  static std::vector<Eigen::MatrixXd> ForwardDynamicsTrajectory(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::MatrixXd& taumat,
      const Eigen::Vector3d& g,
      const Eigen::MatrixXd& Ftipmat,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist,
      double dt,
      int intRes);
};
}  // namespace mymr
