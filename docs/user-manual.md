# My Modern Robotics C++ User Manual

## Overview

My Modern Robotics C++ is a class-based C++ API for the core algorithms from
the Modern Robotics textbook. The library keeps the original function names
and signatures, but organizes them into focused classes for clarity.

### Class map

- `mymr::Tools` - shared math helpers, SE(3)/so(3), Jacobians
- `mymr::FK` - forward kinematics (body/space)
- `mymr::IK` - inverse kinematics (body/space)
- `mymr::InverseDynamics` - inverse dynamics solver
- `mymr::Dynamics` - mass matrix, forces, forward dynamics, trajectories
- `mymr::Trajectory` - time scaling and trajectory generation
- `mymr::RobotControl` - computed torque control and simulation
- `mymr::MotionPlanning` - placeholder for future planning utilities

## Prerequisites

- CMake >= 3.16
- C++17 compiler
- vcpkg (for Eigen and GTest)

## Install vcpkg (local)

This repo expects vcpkg to live in the workspace tools folder.

```sh
git clone https://github.com/microsoft/vcpkg.git "tools/vcpkg"
"tools/vcpkg/bootstrap-vcpkg.sh"
```

Optional environment variables:

```sh
export VCPKG_ROOT="/chalmers/users/jingyang/Documents/workspace/tools/vcpkg"
export PATH="$VCPKG_ROOT:$PATH"
```

## Build and test

This project uses vcpkg manifest mode (`vcpkg.json`) and CMake presets.

```sh
export VCPKG_ROOT="/chalmers/users/jingyang/Documents/workspace/tools/vcpkg"
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## Directory layout

- `include/my_modern_robotics/` - public headers
- `src/` - implementation
- `tests/` - gtest unit tests
- `docs/` - documentation (this manual)

## API conventions

- All functions are static methods on the domain classes.
- Types are Eigen matrices/vectors, matching the reference API.
- Input units follow the Modern Robotics textbook.
- Tolerances for IK are `eomg` (rotational) and `ev` (translational).

## Usage examples

### Tools

```cpp
#include <Eigen/Dense>
#include <my_modern_robotics/tools.h>

Eigen::Vector3d omega(0.0, 0.0, 1.0);
Eigen::Matrix3d so3 = mymr::Tools::VecToso3(omega);
Eigen::Matrix3d R = mymr::Tools::MatrixExp3(so3);
```

### Forward kinematics

```cpp
#include <Eigen/Dense>
#include <my_modern_robotics/fk.h>

Eigen::MatrixXd M = Eigen::MatrixXd::Identity(4, 4);
Eigen::MatrixXd Slist(6, 1);
Eigen::VectorXd thetalist(1);
Eigen::MatrixXd T = mymr::FK::FKinSpace(M, Slist, thetalist);
```

### Inverse kinematics

```cpp
#include <Eigen/Dense>
#include <my_modern_robotics/ik.h>

Eigen::MatrixXd M = Eigen::MatrixXd::Identity(4, 4);
Eigen::MatrixXd Blist(6, 1);
Eigen::MatrixXd T = Eigen::MatrixXd::Identity(4, 4);
Eigen::VectorXd thetalist(1);
bool success = mymr::IK::IKinBody(Blist, M, T, thetalist, 1e-3, 1e-3);
```

### Dynamics

```cpp
#include <Eigen/Dense>
#include <vector>
#include <my_modern_robotics/dynamics.h>

Eigen::VectorXd thetalist(1);
std::vector<Eigen::MatrixXd> Mlist;
std::vector<Eigen::MatrixXd> Glist;
Eigen::MatrixXd Slist(6, 1);
Eigen::MatrixXd M = mymr::Dynamics::MassMatrix(thetalist, Mlist, Glist, Slist);
```

### Inverse dynamics

```cpp
#include <Eigen/Dense>
#include <vector>
#include <my_modern_robotics/inverse_dynamics.h>

Eigen::VectorXd thetalist(1);
Eigen::VectorXd dthetalist(1);
Eigen::VectorXd ddthetalist(1);
Eigen::Vector3d g(0.0, 0.0, -9.81);
Eigen::VectorXd Ftip = Eigen::VectorXd::Zero(6);
std::vector<Eigen::MatrixXd> Mlist;
std::vector<Eigen::MatrixXd> Glist;
Eigen::MatrixXd Slist(6, 1);
Eigen::VectorXd tau = mymr::InverseDynamics::Compute(
    thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist);
```

### Trajectory

```cpp
#include <Eigen/Dense>
#include <my_modern_robotics/trajectory.h>

Eigen::VectorXd thetastart(1);
Eigen::VectorXd thetaend(1);
Eigen::MatrixXd traj = mymr::Trajectory::JointTrajectory(
    thetastart, thetaend, 2.0, 100, 5);
```

### Robot control

```cpp
#include <Eigen/Dense>
#include <vector>
#include <my_modern_robotics/robot_control.h>

Eigen::VectorXd thetalist(1);
Eigen::VectorXd dthetalist(1);
Eigen::VectorXd eint(1);
Eigen::VectorXd thetalistd(1);
Eigen::VectorXd dthetalistd(1);
Eigen::VectorXd ddthetalistd(1);
Eigen::Vector3d g(0.0, 0.0, -9.81);
std::vector<Eigen::MatrixXd> Mlist;
std::vector<Eigen::MatrixXd> Glist;
Eigen::MatrixXd Slist(6, 1);
Eigen::VectorXd tau = mymr::RobotControl::ComputedTorque(
    thetalist,
    dthetalist,
    eint,
    thetalistd,
    dthetalistd,
    ddthetalistd,
    g,
    Mlist,
    Glist,
    Slist,
    1.0,
    0.0,
    0.1);
```

## Testing policy

- All classes have unit tests under `tests/`.
- Accuracy tests use reference values from the Modern Robotics examples.
- Surface tests validate shapes, finiteness, and basic invariants.

To run all tests:

```sh
ctest --preset default
```

## Numerical considerations

- Many routines involve matrix logarithms and exponentials. Expect small
  floating point differences compared to the reference implementation.
- Near singular configurations can cause large Jacobian condition numbers.
- IK convergence depends on the initial guess and tolerances.

## Troubleshooting

**CMake cannot find vcpkg toolchain**
- Confirm `VCPKG_ROOT` is set correctly.

**Eigen or GTest not found**
- Ensure `vcpkg.json` is present and run `cmake --preset default` again.

**IK fails to converge**
- Try a different initial guess or loosen `eomg`/`ev` slightly.

## Versioning and stability

- This project follows semantic versioning.
- Public API names match the reference Modern Robotics implementation.
- MotionPlanning is reserved for future additions.
