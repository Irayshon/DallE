#pragma once

#include <Eigen/Dense>
#include <vector>

namespace WallE {
/**
 * @brief Robot control utilities (computed torque and simulation).
 */
class RobotControl {
 public:
  /**
   * @brief Compute joint torques via computed torque control.
   * @param thetalist Eigen::VectorXd current joint angles.
   * @param dthetalist Eigen::VectorXd current joint velocities.
   * @param eint Eigen::VectorXd integrated position error.
   * @param thetalistd Eigen::VectorXd desired joint angles.
   * @param dthetalistd Eigen::VectorXd desired joint velocities.
   * @param ddthetalistd Eigen::VectorXd desired joint accelerations.
   * @param g Eigen::Vector3d gravity vector (model).
   * @param Mlist std::vector<Eigen::MatrixXd> link transforms (model).
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias (model).
   * @param Slist Eigen::MatrixXd screw axes in space frame.
   * @param Kp double proportional gain.
   * @param Ki double integral gain.
   * @param Kd double derivative gain.
   * @return Eigen::VectorXd n-vector of joint torques.
   */
  static Eigen::VectorXd ComputedTorque(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::VectorXd& eint,
      const Eigen::VectorXd& thetalistd,
      const Eigen::VectorXd& dthetalistd,
      const Eigen::VectorXd& ddthetalistd,
      const Eigen::Vector3d& g,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist,
      double Kp,
      double Ki,
      double Kd);

  /**
   * @brief Simulate computed torque control over a trajectory.
   * @param thetalist Eigen::VectorXd initial joint angles.
   * @param dthetalist Eigen::VectorXd initial joint velocities.
   * @param g Eigen::Vector3d gravity vector (true model).
   * @param Ftipmat Eigen::MatrixXd end-effector wrench trajectory (rows).
   * @param Mlist std::vector<Eigen::MatrixXd> link transforms (true model).
   * @param Glist std::vector<Eigen::MatrixXd> spatial inertias (true model).
   * @param Slist Eigen::MatrixXd screw axes in space frame.
   * @param thetamatd Eigen::MatrixXd desired joint angle trajectory (rows).
   * @param dthetamatd Eigen::MatrixXd desired joint velocity trajectory (rows).
   * @param ddthetamatd Eigen::MatrixXd desired joint acceleration trajectory (rows).
   * @param gtilde Eigen::Vector3d gravity vector (estimated model).
   * @param Mtildelist std::vector<Eigen::MatrixXd> transforms (estimated model).
   * @param Gtildelist std::vector<Eigen::MatrixXd> inertias (estimated model).
   * @param Kp double proportional gain.
   * @param Ki double integral gain.
   * @param Kd double derivative gain.
   * @param dt double time step between rows.
   * @param intRes int integration resolution per step.
   * @return std::vector<Eigen::MatrixXd> tau trajectory and joint angles.
   */
  static std::vector<Eigen::MatrixXd> SimulateControl(
      const Eigen::VectorXd& thetalist,
      const Eigen::VectorXd& dthetalist,
      const Eigen::Vector3d& g,
      const Eigen::MatrixXd& Ftipmat,
      const std::vector<Eigen::MatrixXd>& Mlist,
      const std::vector<Eigen::MatrixXd>& Glist,
      const Eigen::MatrixXd& Slist,
      const Eigen::MatrixXd& thetamatd,
      const Eigen::MatrixXd& dthetamatd,
      const Eigen::MatrixXd& ddthetamatd,
      const Eigen::Vector3d& gtilde,
      const std::vector<Eigen::MatrixXd>& Mtildelist,
      const std::vector<Eigen::MatrixXd>& Gtildelist,
      double Kp,
      double Ki,
      double Kd,
      double dt,
      int intRes);
};
}  // namespace WallE
