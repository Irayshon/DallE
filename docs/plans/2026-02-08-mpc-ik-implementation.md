# MPC-IK Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add trajectory-tracking MPC-based Inverse Kinematics (MPCIKinBody, MPCIKinSpace) to the WallE library using OSQP.

**Architecture:** Receding-horizon QP solver built on existing FK/Jacobian/Tools functions. Decision variables are joint velocity increments over a prediction horizon. QP cost = tracking error + velocity regularization + acceleration regularization. Six constraint types linearized at each step. OSQP via osqp-eigen solves each QP.

**Tech Stack:** C++17, Eigen3, osqp-eigen (via vcpkg), GTest

**Worktree:** `/chalmers/users/jingyang/Documents/workspace/modern/WallE/.worktrees/mpc-ik` (branch `feature/mpc-ik`)

**Build commands:**
```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

**Design doc:** `docs/plans/2026-02-08-mpc-ik-design.md`

---

### Task 1: Add osqp-eigen Dependency

**Files:**
- Modify: `vcpkg.json`
- Modify: `CMakeLists.txt`

**Step 1: Update vcpkg.json**

Change `vcpkg.json` to:
```json
{
  "name": "walle",
  "version-string": "0.1.0",
  "dependencies": [
    "eigen3",
    "gtest",
    "osqp-eigen"
  ]
}
```

Note: also fix the package name from `my-modern-robotics-cpp` to `walle`.

**Step 2: Update CMakeLists.txt**

After `find_package(Eigen3 CONFIG REQUIRED)` (line 10), add:
```cmake
find_package(OsqpEigen REQUIRED)
```

Change `target_link_libraries(WallE PUBLIC Eigen3::Eigen)` (line 27) to:
```cmake
target_link_libraries(WallE PUBLIC Eigen3::Eigen OsqpEigen::OsqpEigen)
```

**Step 3: Verify build**

Run:
```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Expected: All 27 existing tests pass. osqp-eigen is installed by vcpkg.

**Step 4: Commit**

```bash
git add vcpkg.json CMakeLists.txt
git commit -m "build: add osqp-eigen dependency for MPC-IK"
```

---

### Task 2: Add API Declarations to ik.h

**Files:**
- Modify: `include/WallE/ik.h`

**Step 1: Add includes and structs**

After `#include <Eigen/Dense>` (line 3), add:
```cpp
#include <vector>
#include <tuple>
```

Before the `IK` class (after line 5, inside namespace WallE), add:
```cpp
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
```

Inside the `IK` class, after the `IKinSpace` declaration (after line 43), add:
```cpp
  /**
   * @brief MPC-based inverse kinematics tracking a trajectory in body frame.
   * @param Blist Eigen::MatrixXd 6xn screw axes in the body frame.
   * @param M Eigen::MatrixXd home configuration of the end-effector.
   * @param T_trajectory std::vector<Eigen::Matrix4d> desired end-effector poses.
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
   * @param T_trajectory std::vector<Eigen::Matrix4d> desired end-effector poses.
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
```

**Step 2: Verify build compiles (declarations only, no implementation yet — will fail at link)**

Run:
```bash
cmake --build --preset default 2>&1 | head -5
```
Expected: Compiles OK (no syntax errors in header). Link errors expected (no implementation yet).

**Step 3: Commit**

```bash
git add include/WallE/ik.h
git commit -m "feat: add MPCIKConfig, MPCIKResult, and MPC-IK method declarations"
```

---

### Task 3: Implement MPC-IK Core Solver

**Files:**
- Modify: `src/ik.cpp`

This is the main implementation task. The structure in `ik.cpp`:

1. Add `#include <OsqpEigen/OsqpEigen.h>` at top
2. Add anonymous namespace helper functions:
   - `ComputeTaskError()` — compute 6-vector task-space error using existing WallE functions
   - `ComputeLinkPositions()` — FK at intermediate joints for collision avoidance
   - `ComputeManipulability()` — sqrt(det(J*J^T)) using existing Tools functions
   - `NumericalGradient()` — finite-difference gradient of a scalar function w.r.t. joint angles
   - `BuildMPCQP()` — construct the full QP matrices (H, g, A, lb, ub) for the horizon
   - `MPCIKSolve()` — the main receding-horizon loop (shared by Body and Space variants)
3. Implement `IK::MPCIKinBody()` — calls MPCIKSolve with body-frame FK/Jacobian lambdas
4. Implement `IK::MPCIKinSpace()` — calls MPCIKSolve with space-frame FK/Jacobian lambdas

**Step 1: Add includes**

At top of `ik.cpp`, after existing includes, add:
```cpp
#include <OsqpEigen/OsqpEigen.h>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>
```

**Step 2: Implement helper functions in anonymous namespace**

After the existing `PseudoInverse` function (line 23), add the following helpers.

`ComputeTaskError`: Given current FK transform and target, compute 6-vector error.
```cpp
Eigen::VectorXd ComputeTaskError(const Eigen::MatrixXd& T_current,
                                 const Eigen::Matrix4d& T_target) {
  return WallE::Tools::se3ToVec(
      WallE::Tools::MatrixLog6(WallE::Tools::TransInv(T_current) * T_target));
}
```

`ComputeManipulability`: Compute sqrt(det(J*J^T)).
```cpp
double ComputeManipulability(const Eigen::MatrixXd& J) {
  if (J.rows() > J.cols()) return 0.0;
  Eigen::MatrixXd JJt = J * J.transpose();
  double det = JJt.determinant();
  return det > 0.0 ? std::sqrt(det) : 0.0;
}
```

`ComputeLinkPositions`: Compute position of each link origin by applying FK up to joint i. Uses the screw axes and FKinBody/FKinSpace pattern but stopping at each joint. For body frame:
```cpp
std::vector<Eigen::Vector3d> ComputeLinkPositions(
    const Eigen::MatrixXd& M,
    const Eigen::MatrixXd& screw_axes,
    const Eigen::VectorXd& thetalist,
    const std::function<Eigen::MatrixXd(const Eigen::MatrixXd&,
                                        const Eigen::MatrixXd&,
                                        const Eigen::VectorXd&)>& fk_func) {
  int n = thetalist.size();
  std::vector<Eigen::Vector3d> positions(n + 1);
  // Base position
  positions[0] = Eigen::Vector3d::Zero();
  // Position of each link by computing FK with first i joints
  for (int i = 1; i <= n; ++i) {
    Eigen::MatrixXd sub_axes = screw_axes.leftCols(i);
    Eigen::VectorXd sub_theta = thetalist.head(i);
    // Use identity as M for sub-chain, get position from transform
    Eigen::MatrixXd T_i = fk_func(M, sub_axes, sub_theta);
    positions[i] = T_i.block<3, 1>(0, 3);
  }
  return positions;
}
```

`BuildMPCQP`: Construct the full QP for the MPC horizon. This is the core function.

Decision variables: x = [Δθ̇₀, Δθ̇₁, ..., Δθ̇_{N-1}] (N*n total)

The QP is: min 0.5 * x^T * H * x + g^T * x, subject to lb <= A*x <= ub.

H is block-diagonal with blocks: `J^T * Q * J * dt^2 + R * dt^2 + S` for each step k.
g contains the gradient terms from linearized tracking error.
A contains the constraint matrix rows for position/velocity/acceleration limits, workspace, collision, and singularity.

```cpp
struct QPData {
  Eigen::SparseMatrix<double> H;
  Eigen::VectorXd g;
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd lb;
  Eigen::VectorXd ub;
  int num_vars;
  int num_constraints;
};

QPData BuildMPCQP(
    int n, int N, double dt,
    const Eigen::VectorXd& theta_current,
    const Eigen::VectorXd& dtheta_current,
    const std::vector<Eigen::MatrixXd>& jacobians,      // Jacobian at each predicted step
    const std::vector<Eigen::VectorXd>& task_errors,     // task-space error at each predicted step
    double w_tracking, double w_velocity, double w_acceleration,
    const WallE::MPCIKConfig& config) {

  QPData qp;
  qp.num_vars = N * n;

  // --- Count constraints ---
  // For each horizon step k = 0..N-1:
  //   - n position limit constraints (theta_k+1 in bounds)
  //   - n velocity limit constraints (dtheta_k+1 in bounds)
  //   - n acceleration limit constraints (ddtheta_k in bounds)
  int n_basic = 3 * n * N;

  // Workspace: 3 constraints per step (x,y,z of EE position)
  bool has_workspace = config.workspace_min.norm() < 1e9 || config.workspace_max.norm() < 1e9;
  int n_workspace = has_workspace ? 3 * N : 0;

  // Collision: one constraint per pair per step
  int n_collision = static_cast<int>(config.collision_pairs.size()) * N;

  // Singularity: one constraint per step
  bool has_singularity = config.manipulability_threshold > 0.0;
  int n_singularity = has_singularity ? N : 0;

  qp.num_constraints = n_basic + n_workspace + n_collision + n_singularity;

  // --- Build H (Hessian) ---
  // H is N*n x N*n block diagonal
  std::vector<Eigen::Triplet<double>> H_triplets;
  H_triplets.reserve(N * n * n * 3); // rough estimate

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6) * w_tracking;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n) * w_velocity;
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n) * w_acceleration;

  for (int k = 0; k < N; ++k) {
    // H_k = J_k^T * Q * J_k * dt^2 + R * dt^2 + S
    const Eigen::MatrixXd& J_k = jacobians[k];
    Eigen::MatrixXd H_k = J_k.transpose() * Q * J_k * dt * dt
                         + R * dt * dt + S;

    // Make symmetric (numerical safety)
    H_k = 0.5 * (H_k + H_k.transpose());

    int offset = k * n;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (std::abs(H_k(i, j)) > 1e-12) {
          H_triplets.emplace_back(offset + i, offset + j, H_k(i, j));
        }
      }
    }
  }

  qp.H.resize(qp.num_vars, qp.num_vars);
  qp.H.setFromTriplets(H_triplets.begin(), H_triplets.end());

  // --- Build g (gradient) ---
  qp.g.resize(qp.num_vars);
  for (int k = 0; k < N; ++k) {
    const Eigen::MatrixXd& J_k = jacobians[k];
    const Eigen::VectorXd& e_k = task_errors[k];
    // g_k = -J_k^T * Q * e_k * dt + R * dtheta_predicted_k * dt^2
    // For simplicity, dtheta_predicted at step k ≈ dtheta_current (linearized)
    qp.g.segment(k * n, n) = -J_k.transpose() * Q * e_k * dt
                            + R * dtheta_current * dt * dt;
  }

  // --- Build A, lb, ub (constraints) ---
  std::vector<Eigen::Triplet<double>> A_triplets;
  qp.lb.resize(qp.num_constraints);
  qp.ub.resize(qp.num_constraints);

  int row = 0;

  // Predict states forward for constraint building
  Eigen::VectorXd theta_pred = theta_current;
  Eigen::VectorXd dtheta_pred = dtheta_current;

  for (int k = 0; k < N; ++k) {
    int col_offset = k * n;

    // --- Position limits: theta_min <= theta_pred + dtheta_pred*dt + Δθ̇_k*dt^2 <= theta_max ---
    if (config.theta_min.size() == n) {
      for (int i = 0; i < n; ++i) {
        // Cumulative effect: sum of all Δθ̇_j for j <= k affects theta at step k+1
        for (int j = 0; j <= k; ++j) {
          double coeff = (k - j + 1) * dt * dt;
          A_triplets.emplace_back(row + i, j * n + i, coeff);
        }
        double theta_pred_k = theta_current(i) + dtheta_current(i) * (k + 1) * dt;
        qp.lb(row + i) = config.theta_min(i) - theta_pred_k;
        qp.ub(row + i) = config.theta_max(i) - theta_pred_k;
      }
    } else {
      // No position limits — set wide bounds
      for (int i = 0; i < n; ++i) {
        A_triplets.emplace_back(row + i, col_offset + i, 1.0);
        qp.lb(row + i) = -1e10;
        qp.ub(row + i) = 1e10;
      }
    }
    row += n;

    // --- Velocity limits: dtheta_min <= dtheta_current + sum(Δθ̇_j, j<=k) <= dtheta_max ---
    if (config.dtheta_min.size() == n) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= k; ++j) {
          A_triplets.emplace_back(row + i, j * n + i, 1.0);
        }
        qp.lb(row + i) = config.dtheta_min(i) - dtheta_current(i);
        qp.ub(row + i) = config.dtheta_max(i) - dtheta_current(i);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        A_triplets.emplace_back(row + i, col_offset + i, 1.0);
        qp.lb(row + i) = -1e10;
        qp.ub(row + i) = 1e10;
      }
    }
    row += n;

    // --- Acceleration limits: ddtheta_min <= Δθ̇_k / dt <= ddtheta_max ---
    if (config.ddtheta_min.size() == n) {
      for (int i = 0; i < n; ++i) {
        A_triplets.emplace_back(row + i, col_offset + i, 1.0 / dt);
        qp.lb(row + i) = config.ddtheta_min(i);
        qp.ub(row + i) = config.ddtheta_max(i);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        A_triplets.emplace_back(row + i, col_offset + i, 1.0);
        qp.lb(row + i) = -1e10;
        qp.ub(row + i) = 1e10;
      }
    }
    row += n;
  }

  // --- Workspace bounds ---
  // (Linearized: p_min <= p_current + Jv * Δθ * dt <= p_max)
  // Jv = rows 3,4,5 of body Jacobian (linear velocity part)
  if (has_workspace) {
    for (int k = 0; k < N; ++k) {
      const Eigen::MatrixXd& J_k = jacobians[k];
      Eigen::MatrixXd Jv = J_k.bottomRows(3); // linear velocity rows
      Eigen::Vector3d p_current = Eigen::Vector3d::Zero(); // updated per step in main loop
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < n; ++j) {
          double coeff = Jv(i, j) * dt;
          if (std::abs(coeff) > 1e-12) {
            A_triplets.emplace_back(row + i, k * n + j, coeff);
          }
        }
        qp.lb(row + i) = config.workspace_min(i) - p_current(i);
        qp.ub(row + i) = config.workspace_max(i) - p_current(i);
      }
      row += 3;
    }
  }

  // --- Collision avoidance ---
  // Placeholder rows — gradient computed in main loop and updated
  for (int k = 0; k < N; ++k) {
    for (size_t c = 0; c < config.collision_pairs.size(); ++c) {
      // Will be filled with finite-difference gradient
      A_triplets.emplace_back(row, k * n, 0.0); // placeholder
      qp.lb(row) = 0.0;
      qp.ub(row) = 1e10;
      ++row;
    }
  }

  // --- Singularity avoidance ---
  if (has_singularity) {
    for (int k = 0; k < N; ++k) {
      // Will be filled with manipulability gradient
      A_triplets.emplace_back(row, k * n, 0.0); // placeholder
      qp.lb(row) = 0.0;
      qp.ub(row) = 1e10;
      ++row;
    }
  }

  qp.A.resize(qp.num_constraints, qp.num_vars);
  qp.A.setFromTriplets(A_triplets.begin(), A_triplets.end());

  return qp;
}
```

**Step 3: Implement MPCIKSolve — the main receding-horizon loop**

```cpp
// Type aliases for FK and Jacobian function pointers
using FKFunc = std::function<Eigen::MatrixXd(
    const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&)>;
using JacFunc = std::function<Eigen::MatrixXd(
    const Eigen::MatrixXd&, const Eigen::VectorXd&)>;

WallE::MPCIKResult MPCIKSolve(
    const Eigen::MatrixXd& screw_axes,
    const Eigen::MatrixXd& M,
    const std::vector<Eigen::Matrix4d>& T_trajectory,
    Eigen::VectorXd& thetalist,
    const WallE::MPCIKConfig& config,
    const FKFunc& fk_func,
    const JacFunc& jac_func) {

  int n = thetalist.size();
  int N = config.horizon;
  double dt = config.dt;

  WallE::MPCIKResult result;
  result.thetalist = thetalist;
  result.dthetalist = Eigen::VectorXd::Zero(n);

  Eigen::VectorXd theta = thetalist;
  Eigen::VectorXd dtheta = Eigen::VectorXd::Zero(n);

  int traj_index = 0;

  for (int iter = 0; iter < config.max_iterations; ++iter) {
    // Current FK and error
    Eigen::MatrixXd T_current = fk_func(M, screw_axes, theta);
    int target_idx = std::min(traj_index, static_cast<int>(T_trajectory.size()) - 1);
    Eigen::VectorXd error = ComputeTaskError(T_current, T_trajectory[target_idx]);

    double omg_err = error.head(3).norm();
    double v_err = error.tail(3).norm();

    // Check convergence
    if (omg_err <= config.eomg && v_err <= config.ev) {
      // If at last trajectory point, we're done
      if (target_idx >= static_cast<int>(T_trajectory.size()) - 1) {
        result.converged = true;
        result.iterations = iter;
        result.final_orientation_error = omg_err;
        result.final_position_error = v_err;
        result.thetalist = theta;
        result.dthetalist = dtheta;
        thetalist = theta;
        return result;
      }
      // Move to next trajectory point
      ++traj_index;
    }

    // Predict Jacobians and errors over horizon
    std::vector<Eigen::MatrixXd> jacobians(N);
    std::vector<Eigen::VectorXd> task_errors(N);
    Eigen::VectorXd theta_pred = theta;
    std::vector<Eigen::Vector3d> ee_positions(N);

    for (int k = 0; k < N; ++k) {
      int ref_idx = std::min(traj_index + k,
                             static_cast<int>(T_trajectory.size()) - 1);
      jacobians[k] = jac_func(screw_axes, theta_pred);
      Eigen::MatrixXd T_pred = fk_func(M, screw_axes, theta_pred);
      task_errors[k] = ComputeTaskError(T_pred, T_trajectory[ref_idx]);
      ee_positions[k] = T_pred.block<3, 1>(0, 3);

      // Predict forward (linearized)
      theta_pred = theta_pred + dtheta * dt;
    }

    // Build QP
    QPData qp = BuildMPCQP(n, N, dt, theta, dtheta, jacobians, task_errors,
                           config.w_tracking, config.w_velocity,
                           config.w_acceleration, config);

    // Update workspace bounds with actual EE positions
    bool has_workspace = config.workspace_min.norm() < 1e9 ||
                         config.workspace_max.norm() < 1e9;
    if (has_workspace) {
      int ws_row_start = 3 * n * N; // after basic constraints
      for (int k = 0; k < N; ++k) {
        for (int i = 0; i < 3; ++i) {
          int row = ws_row_start + k * 3 + i;
          qp.lb(row) = config.workspace_min(i) - ee_positions[k](i);
          qp.ub(row) = config.workspace_max(i) - ee_positions[k](i);
        }
      }
    }

    // Update collision constraints with finite-difference gradients
    int coll_row_start = 3 * n * N + (has_workspace ? 3 * N : 0);
    for (int k = 0; k < N; ++k) {
      for (size_t c = 0; c < config.collision_pairs.size(); ++c) {
        auto [link_i, link_j, d_min] = config.collision_pairs[c];
        int row = coll_row_start + k * static_cast<int>(config.collision_pairs.size()) + static_cast<int>(c);

        // Compute current distance between link origins
        auto positions = ComputeLinkPositions(M, screw_axes, theta, fk_func);
        int li = std::min(link_i, static_cast<int>(positions.size()) - 1);
        int lj = std::min(link_j, static_cast<int>(positions.size()) - 1);
        double d_current = (positions[li] - positions[lj]).norm();

        // Finite-difference gradient of distance w.r.t. theta
        double eps = 1e-6;
        for (int j = 0; j < n; ++j) {
          Eigen::VectorXd theta_perturbed = theta;
          theta_perturbed(j) += eps;
          auto pos_perturbed = ComputeLinkPositions(M, screw_axes, theta_perturbed, fk_func);
          double d_perturbed = (pos_perturbed[li] - pos_perturbed[lj]).norm();
          double grad = (d_perturbed - d_current) / eps;
          qp.A.coeffRef(row, k * n + j) = grad;
        }
        qp.lb(row) = d_min - d_current;
        qp.ub(row) = 1e10;
      }
    }

    // Update singularity constraints
    bool has_singularity = config.manipulability_threshold > 0.0;
    if (has_singularity) {
      int sing_row_start = coll_row_start +
          static_cast<int>(config.collision_pairs.size()) * N;
      double w_current = ComputeManipulability(jacobians[0]);
      double eps = 1e-6;

      for (int k = 0; k < N; ++k) {
        int row = sing_row_start + k;
        // Finite-difference gradient of manipulability
        for (int j = 0; j < n; ++j) {
          Eigen::VectorXd theta_perturbed = theta;
          theta_perturbed(j) += eps;
          Eigen::MatrixXd J_perturbed = jac_func(screw_axes, theta_perturbed);
          double w_perturbed = ComputeManipulability(J_perturbed);
          double grad = (w_perturbed - w_current) / eps;
          qp.A.coeffRef(row, k * n + j) = grad;
        }
        qp.lb(row) = config.manipulability_threshold - w_current;
        qp.ub(row) = 1e10;
      }
    }

    // Solve QP with OSQP
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.settings()->setPolish(true);

    solver.data()->setNumberOfVariables(qp.num_vars);
    solver.data()->setNumberOfConstraints(qp.num_constraints);

    if (!solver.data()->setHessianMatrix(qp.H)) break;
    if (!solver.data()->setGradient(qp.g)) break;
    if (!solver.data()->setLinearConstraintsMatrix(qp.A)) break;
    if (!solver.data()->setLowerBound(qp.lb)) break;
    if (!solver.data()->setUpperBound(qp.ub)) break;

    if (!solver.initSolver()) break;

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) break;

    Eigen::VectorXd solution = solver.getSolution();

    // Apply first control step
    Eigen::VectorXd ddtheta_0 = solution.head(n);
    dtheta += ddtheta_0;
    theta += dtheta * dt;

    result.iterations = iter + 1;
    result.final_orientation_error = omg_err;
    result.final_position_error = v_err;
  }

  // Did not converge
  result.converged = false;
  result.thetalist = theta;
  result.dthetalist = dtheta;
  thetalist = theta;
  return result;
}
```

**Step 4: Implement IK::MPCIKinBody and IK::MPCIKinSpace**

```cpp
namespace WallE {

MPCIKResult IK::MPCIKinBody(
    const Eigen::MatrixXd& Blist,
    const Eigen::MatrixXd& M,
    const std::vector<Eigen::Matrix4d>& T_trajectory,
    Eigen::VectorXd& thetalist,
    const MPCIKConfig& config) {
  auto fk = [](const Eigen::MatrixXd& M_arg,
               const Eigen::MatrixXd& axes,
               const Eigen::VectorXd& theta) {
    return FK::FKinBody(M_arg, axes, theta);
  };
  auto jac = [](const Eigen::MatrixXd& axes,
                const Eigen::VectorXd& theta) {
    return Tools::JacobianBody(axes, theta);
  };
  return MPCIKSolve(Blist, M, T_trajectory, thetalist, config, fk, jac);
}

MPCIKResult IK::MPCIKinSpace(
    const Eigen::MatrixXd& Slist,
    const Eigen::MatrixXd& M,
    const std::vector<Eigen::Matrix4d>& T_trajectory,
    Eigen::VectorXd& thetalist,
    const MPCIKConfig& config) {
  auto fk = [](const Eigen::MatrixXd& M_arg,
               const Eigen::MatrixXd& axes,
               const Eigen::VectorXd& theta) {
    return FK::FKinSpace(M_arg, axes, theta);
  };
  auto jac = [](const Eigen::MatrixXd& axes,
                const Eigen::VectorXd& theta) {
    return Tools::JacobianSpace(axes, theta);
  };
  return MPCIKSolve(Slist, M, T_trajectory, thetalist, config, fk, jac);
}

}  // namespace WallE
```

**Step 5: Build and verify existing tests still pass**

Run:
```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Expected: All 27 tests pass. No linker errors.

**Step 6: Commit**

```bash
git add src/ik.cpp
git commit -m "feat: implement MPC-IK core solver with OSQP"
```

---

### Task 4: Write Tests

**Files:**
- Modify: `tests/ik_test.cpp`

Add 9 new test cases after the existing `IKTest.PlanarTwoLinkTarget` test. All tests follow the same pattern: set up robot kinematics, create trajectory, configure MPC-IK, solve, verify results.

**Test 1: MPCPlanarTwoLinkTrajectory** — 2-DOF planar arm tracks a 3-point circular arc.
**Test 2: MPCThreeDOFJointLimits** — 3-DOF arm with tight joint limits.
**Test 3: MPCSingularityAvoidance** — Start near singularity, verify no divergence.
**Test 4: MPCWorkspaceBounds** — Set workspace box, verify EE stays in bounds.
**Test 5: MPCSinglePoseConvergence** — Same test case as existing IK, single-pose trajectory.
**Test 6: MPCBodySpaceConsistency** — Same problem, both frames, same result.
**Test 7: MPCNonConvergence** — Unreachable target returns converged=false.
**Test 8: MPCSevenDOFArm** — Franka-like 7-DOF arm with trajectory and constraints.
**Test 9: MPCQuadrupedLeg** — Go2-like 3-DOF leg tracking foot swing trajectory.

Full test code will be provided in implementation (see design doc for test descriptions).

**Step 1: Write all 9 tests**

(Full code provided during implementation — each test is ~30-60 lines)

**Step 2: Build and run**

```bash
cmake --build --preset default
ctest --preset default
```

Expected: All 27 + 9 = 36 tests pass.

**Step 3: Commit**

```bash
git add tests/ik_test.cpp
git commit -m "test: add 9 MPC-IK test cases including 7-DOF arm and Go2 leg"
```

---

### Task 5: Final Verification

**Step 1: Clean build from scratch**

```bash
rm -rf build/
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Expected: All 36 tests pass.

**Step 2: Verify no changes to existing API**

Existing IKinBody/IKinSpace tests must pass unchanged.

**Step 3: Review for code quality**

- No `as any` / `@ts-ignore` equivalent
- No empty catch blocks
- Proper error handling
- Doxygen comments on public API
