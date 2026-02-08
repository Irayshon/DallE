#pragma once

#include <Eigen/Dense>
#include <vector>

namespace WallE {
/**
 * @brief Forward dynamics and force-related helpers.
 */
class Dynamics {
 public:
  /**
   * @brief Compute the joint-space mass matrix.
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::MatrixXd n x n mass matrix.
   */
  static Eigen::MatrixXd MassMatrix(
      const Eigen::VectorXd& thetalist,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute velocity quadratic forces (Coriolis/centripetal).
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param dthetalist Eigen::VectorXd n-vector of joint velocities.
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::VectorXd n-vector of joint forces/torques.
   */
  static Eigen::VectorXd VelQuadraticForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute gravity forces at the joints.
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param g Eigen::Vector3d gravity vector.
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::VectorXd n-vector of joint forces/torques.
   */
  static Eigen::VectorXd GravityForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::Vector3d& g,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute joint forces from an end-effector wrench.
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param Ftip Eigen::VectorXd spatial force at the end-effector.
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::VectorXd n-vector of joint forces/torques.
   */
  static Eigen::VectorXd EndEffectorForces(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& Ftip,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist);

  /**
   * @brief Compute forward dynamics (joint accelerations).
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param dthetalist Eigen::VectorXd n-vector of joint velocities.
   * @param taulist Eigen::VectorXd n-vector of joint forces/torques.
   * @param g Eigen::Vector3d gravity vector.
   * @param Ftip Eigen::VectorXd spatial force at the end-effector.
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::VectorXd n-vector of joint accelerations.
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
   * @param thetalist Eigen::VectorXd n-vector of joint angles.
   * @param dthetalist Eigen::VectorXd n-vector of joint velocities.
   * @param ddthetalist Eigen::VectorXd n-vector of joint accelerations.
   * @param dt double time step.
   * @return std::vector<Eigen::VectorXd> updated theta and dtheta.
   */
  static std::vector<Eigen::VectorXd> EulerStep(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& ddthetalist,
      double dt);

  /**
   * @brief Compute inverse dynamics for a joint-space trajectory.
   * @param thetamat Eigen::MatrixXd joint positions (rows).
   * @param dthetamat Eigen::MatrixXd joint velocities (rows).
   * @param ddthetamat Eigen::MatrixXd joint accelerations (rows).
   * @param g Eigen::Vector3d gravity vector.
   * @param Ftipmat Eigen::MatrixXd end-effector wrenches (rows).
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @return Eigen::MatrixXd joint torques over time (rows).
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
   * @param thetalist Eigen::VectorXd initial joint angles.
   * @param dthetalist Eigen::VectorXd initial joint velocities.
   * @param taumat Eigen::MatrixXd joint torque trajectory (rows).
   * @param g Eigen::Vector3d gravity vector.
   * @param Ftipmat Eigen::MatrixXd end-effector wrenches (rows).
   * @param Mlist std::vector<Eigen::MatrixXd> link frames at home.
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @param dt double time step between rows of taumat.
   * @param intRes int integration resolution per time step.
   * @return std::vector<Eigen::MatrixXd> joint positions and velocities.
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
}  // namespace WallE
