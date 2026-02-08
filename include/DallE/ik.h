#pragma once

#include <Eigen/Dense>

#include <tuple>
#include <vector>

namespace DallE {

/// Configuration for MPC-based IK solver.
struct MPCIKConfig {
  /// Prediction horizon length (number of steps).
  int horizon = 10;
  /// Timestep in seconds.
  double dt = 0.01;
  /// Maximum number of MPC iterations before giving up.
  int max_iterations = 100;

  /// Weight for task-space tracking error (Q diagonal value).
  double w_tracking = 100.0;
  /// Weight for joint velocity regularization (R diagonal value).
  double w_velocity = 1.0;
  /// Weight for joint acceleration regularization (S diagonal value).
  double w_acceleration = 0.1;

  /// Orientation error tolerance for convergence.
  double eomg = 1e-3;
  /// Position error tolerance for convergence.
  double ev = 1e-3;

  /// Joint position lower bounds (n-vector).
  Eigen::VectorXd theta_min;
  /// Joint position upper bounds (n-vector).
  Eigen::VectorXd theta_max;
  /// Joint velocity lower bounds (n-vector).
  Eigen::VectorXd dtheta_min;
  /// Joint velocity upper bounds (n-vector).
  Eigen::VectorXd dtheta_max;
  /// Joint acceleration lower bounds (n-vector).
  Eigen::VectorXd ddtheta_min;
  /// Joint acceleration upper bounds (n-vector).
  Eigen::VectorXd ddtheta_max;

  /// End-effector workspace position lower bounds.
  Eigen::Vector3d workspace_min = Eigen::Vector3d::Constant(-1e10);
  /// End-effector workspace position upper bounds.
  Eigen::Vector3d workspace_max = Eigen::Vector3d::Constant(1e10);

  /// Minimum manipulability index for singularity avoidance.
  double manipulability_threshold = 0.01;

  /// Self-collision pairs: {link_i, link_j, min_distance}.
  std::vector<std::tuple<int, int, double>> collision_pairs;
};

/// Result from MPC-IK solver.
struct MPCIKResult {
  /// Whether the solver converged within tolerances.
  bool converged = false;
  /// Number of MPC iterations performed.
  int iterations = 0;
  /// Final position error norm.
  double final_position_error = 0.0;
  /// Final orientation error norm.
  double final_orientation_error = 0.0;
  /// Final joint configuration.
  Eigen::VectorXd thetalist;
  /// Final joint velocities.
  Eigen::VectorXd dthetalist;
};

/**
 * @brief Inverse kinematics functions.
 */
class IK {
 public:
  /**
   * @brief Compute inverse kinematics in the body frame.
   * @param Blist Eigen::MatrixXd 6xn screw axes in the body frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param T Eigen::MatrixXd desired end-effector configuration.
   * @param thetalist Eigen::VectorXd initial guess; updated on success.
   * @param eomg double orientation error tolerance.
   * @param ev double position error tolerance.
   * @return bool true if solution converged within tolerances.
   */
  static bool IKinBody(const Eigen::MatrixXd& Blist,
                       const Eigen::MatrixXd& M,
                       const Eigen::MatrixXd& T,
                       Eigen::VectorXd& thetalist,
                       double eomg,
                       double ev);

  /**
   * @brief Compute inverse kinematics in the space frame.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param T Eigen::MatrixXd desired end-effector configuration.
   * @param thetalist Eigen::VectorXd initial guess; updated on success.
   * @param eomg double orientation error tolerance.
   * @param ev double position error tolerance.
   * @return bool true if solution converged within tolerances.
   */
  static bool IKinSpace(const Eigen::MatrixXd& Slist,
                        const Eigen::MatrixXd& M,
                        const Eigen::MatrixXd& T,
                        Eigen::VectorXd& thetalist,
                        double eomg,
                        double ev);

  /**
   * @brief MPC-based inverse kinematics tracking a trajectory in body frame.
   * @param Blist Eigen::MatrixXd 6xn screw axes in the body frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param T_trajectory desired end-effector poses along the trajectory.
   * @param thetalist Eigen::VectorXd initial joint config; updated on success.
   * @param config MPCIKConfig solver parameters and constraints.
   * @return MPCIKResult convergence status, final errors, and solution.
   */
  static MPCIKResult MPCIKinBody(
      const Eigen::MatrixXd& Blist,
      const Eigen::MatrixXd& M,
      const std::vector<Eigen::Matrix4d>& T_trajectory,
      Eigen::VectorXd& thetalist,
      const MPCIKConfig& config);

  /**
   * @brief MPC-based inverse kinematics tracking a trajectory in space frame.
   * @param Slist Eigen::MatrixXd 6xn screw axes in the space frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param T_trajectory desired end-effector poses along the trajectory.
   * @param thetalist Eigen::VectorXd initial joint config; updated on success.
   * @param config MPCIKConfig solver parameters and constraints.
   * @return MPCIKResult convergence status, final errors, and solution.
   */
  static MPCIKResult MPCIKinSpace(
      const Eigen::MatrixXd& Slist,
      const Eigen::MatrixXd& M,
      const std::vector<Eigen::Matrix4d>& T_trajectory,
      Eigen::VectorXd& thetalist,
      const MPCIKConfig& config);
};
}  // namespace DallE
