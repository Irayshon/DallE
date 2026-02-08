# MPC-IK Design

## Overview

Add trajectory-tracking MPC-based Inverse Kinematics to the WallE library. The solver uses OSQP (via osqp-eigen) to solve a QP at each control step over a receding prediction horizon. It enforces six constraint types: joint position/velocity/acceleration limits, self-collision avoidance, workspace bounds, and singularity avoidance.

New methods (`MPCIKinBody`, `MPCIKinSpace`) live alongside the existing Newton-Raphson IK in `ik.h`/`ik.cpp`.

## Problem Formulation

### Decision Variables

Joint velocity increments `Δθ̇ₖ` for `k = 0..N-1`, where `N` is the prediction horizon. Each is an n-vector (n = number of joints), giving `N×n` total decision variables.

### State Propagation (Linearized)

```
θₖ₊₁ = θₖ + θ̇ₖ · dt
θ̇ₖ₊₁ = θ̇ₖ + Δθ̇ₖ
eₖ   = se3ToVec(MatrixLog6(TransInv(FKin(θₖ)) · T_ref(k)))
eₖ₊₁ ≈ eₖ - J(θₖ) · θ̇ₖ · dt
```

### Cost Function

```
min  Σₖ ||eₖ||²_Q  +  Σₖ ||θ̇ₖ||²_R  +  Σₖ ||Δθ̇ₖ||²_S
```

- `Q` (6×6): task-space tracking weight
- `R` (n×n): joint velocity regularization
- `S` (n×n): joint acceleration regularization

### Constraints

| # | Constraint | Linear Form |
|---|-----------|-------------|
| 1 | Joint position limits | `θ_min ≤ θₖ ≤ θ_max` |
| 2 | Joint velocity limits | `θ̇_min ≤ θ̇ₖ ≤ θ̇_max` |
| 3 | Joint acceleration limits | `θ̈_min ≤ Δθ̇ₖ/dt ≤ θ̈_max` |
| 4 | Self-collision avoidance | `∂d/∂θ · Δθ ≥ d_min - d(θ_current)` (link origin distances) |
| 5 | Workspace bounds | `p_min ≤ p(θ) + Jv·Δθ·dt ≤ p_max` |
| 6 | Singularity avoidance | `∂w/∂θ · Δθ ≥ ε - w(θ_current)` where `w = √det(JJᵀ)` |

Constraints 4-6 are linearized at each MPC step. Self-collision and singularity gradients are computed via finite differences.

### Receding Horizon

At each MPC step, solve the QP, apply only the first `Δθ̇₀`, advance state, shift the horizon window along `T_trajectory`, and re-solve.

## API

### Configuration

```cpp
struct MPCIKConfig {
  int horizon = 10;
  double dt = 0.01;
  int max_iterations = 100;

  double w_tracking = 100.0;
  double w_velocity = 1.0;
  double w_acceleration = 0.1;

  double eomg = 1e-3;
  double ev = 1e-3;

  Eigen::VectorXd theta_min, theta_max;
  Eigen::VectorXd dtheta_min, dtheta_max;
  Eigen::VectorXd ddtheta_min, ddtheta_max;

  Eigen::Vector3d workspace_min = Eigen::Vector3d::Constant(-1e10);
  Eigen::Vector3d workspace_max = Eigen::Vector3d::Constant(1e10);

  double manipulability_threshold = 0.01;

  std::vector<std::tuple<int, int, double>> collision_pairs;
};
```

### Result

```cpp
struct MPCIKResult {
  bool converged = false;
  int iterations = 0;
  double final_position_error = 0.0;
  double final_orientation_error = 0.0;
  Eigen::VectorXd thetalist;
  Eigen::VectorXd dthetalist;
};
```

### Methods

```cpp
class IK {
 public:
  // Existing (unchanged)
  static bool IKinBody(...);
  static bool IKinSpace(...);

  // New MPC-IK
  static MPCIKResult MPCIKinBody(
      const Eigen::MatrixXd& Blist,
      const Eigen::MatrixXd& M,
      const std::vector<Eigen::Matrix4d>& T_trajectory,
      Eigen::VectorXd& thetalist,
      const MPCIKConfig& config);

  static MPCIKResult MPCIKinSpace(
      const Eigen::MatrixXd& Slist,
      const Eigen::MatrixXd& M,
      const std::vector<Eigen::Matrix4d>& T_trajectory,
      Eigen::VectorXd& thetalist,
      const MPCIKConfig& config);
};
```

## Implementation Architecture

```
MPCIKinBody / MPCIKinSpace
  └── MPCIKSolve (private, shared core)
       ├── BuildQPMatrices()       — H, g, A, lb, ub
       ├── LinearizeConstraints()  — collision, singularity, workspace
       └── Receding horizon loop:
            1. FK + Jacobian (WallE::FK, WallE::Tools)
            2. Task-space error (MatrixLog6 + se3ToVec)
            3. Build QP for horizon N
            4. Solve via OsqpEigen
            5. Apply Δθ̇₀: θ̇ += Δθ̇₀, θ += θ̇·dt
            6. Convergence check (eomg, ev)
            7. Shift horizon window
```

Self-collision uses link-origin distances (no geometry model). The library already provides FK at intermediate frames for computing link positions.

## Dependencies

- `osqp-eigen` (via vcpkg) — C++ wrapper for OSQP with Eigen types
- `osqp` — pulled in as dependency of osqp-eigen

CMake additions:
```cmake
find_package(OsqpEigen REQUIRED)
target_link_libraries(WallE PUBLIC OsqpEigen::OsqpEigen)
```

## Testing

Nine test cases (all in `tests/ik_test.cpp`):

1. Planar 2-link trajectory tracking (circular arc)
2. 3-DOF arm with joint limits
3. Singularity avoidance (near-singular start)
4. Workspace bounds enforcement
5. Convergence on existing IK test cases (single-pose trajectory)
6. Body vs Space frame consistency
7. Non-convergence on unreachable target
8. 7-DOF arm (Franka-like kinematics, redundancy resolution)
9. Quadruped single leg (Go2-like 3-DOF HAA-HFE-KFE, foot trajectory tracking)

Tests 8-9 use hardcoded screw axes and joint limits from datasheets. No Gazebo.
