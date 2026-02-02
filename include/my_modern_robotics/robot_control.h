#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mymr {
/**
 * @brief Robot control utilities (computed torque and simulation).
 */
class RobotControl {
 public:
  /**
   * @brief Compute joint torques via computed torque control.
   * @param thetalist Current joint angles.
   * @param dthetalist Current joint velocities.
   * @param eint Integrated position error.
   * @param thetalistd Desired joint angles.
   * @param dthetalistd Desired joint velocities.
   * @param ddthetalistd Desired joint accelerations.
   * @param g Gravity vector (model).
   * @param Mlist Link transforms at home (model).
   * @param Glist Spatial inertias (model).
   * @param Slist Screw axes in space frame.
   * @param Kp Proportional gain.
   * @param Ki Integral gain.
   * @param Kd Derivative gain.
   * @return n-vector of joint torques.
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
   * @param thetalist Initial joint angles.
   * @param dthetalist Initial joint velocities.
   * @param g Gravity vector (true model).
   * @param Ftipmat End-effector wrench trajectory (rows).
   * @param Mlist Link transforms at home (true model).
   * @param Glist Spatial inertias (true model).
   * @param Slist Screw axes in space frame.
   * @param thetamatd Desired joint angle trajectory (rows).
   * @param dthetamatd Desired joint velocity trajectory (rows).
   * @param ddthetamatd Desired joint acceleration trajectory (rows).
   * @param gtilde Gravity vector used by controller (estimated model).
   * @param Mtildelist Link transforms for controller model.
   * @param Gtildelist Spatial inertias for controller model.
   * @param Kp Proportional gain.
   * @param Ki Integral gain.
   * @param Kd Derivative gain.
   * @param dt Time step between rows.
   * @param intRes Integration resolution per step.
   * @return Vector of two matrices: tau trajectory and joint angles.
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
}  // namespace mymr
