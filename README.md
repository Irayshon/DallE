# My Modern Robotics C++

This project provides a class-based C++ API for core Modern Robotics routines.
The API is split into focused classes so you can include only what you need.

For a full guide, see the user manual: `docs/user-manual.md`.

## Class split

- `mymr::Tools` - shared math helpers, SE(3)/so(3) utilities, Jacobians
- `mymr::FK` - forward kinematics in body or space frames
- `mymr::IK` - inverse kinematics in body or space frames
- `mymr::InverseDynamics` - inverse dynamics solver
- `mymr::Dynamics` - mass matrix, forces, forward dynamics, trajectories
- `mymr::Trajectory` - time scaling and trajectory generation
- `mymr::RobotControl` - computed torque control and simulation helpers
- `mymr::MotionPlanning` - placeholder for upcoming planning utilities

## Usage examples

### Tools

```cpp
#include <Eigen/Dense>
#include <my_modern_robotics/tools.h>

Eigen::Vector3d omega(0.0, 0.0, 1.0);
Eigen::Matrix3d so3 = mymr::Tools::VecToso3(omega);
Eigen::Matrix3d R = mymr::Tools::MatrixExp3(so3);
```

### FK

```cpp
#include <Eigen/Dense>
#include <my_modern_robotics/fk.h>

Eigen::MatrixXd M = Eigen::MatrixXd::Identity(4, 4);
Eigen::MatrixXd Slist(6, 1);
Eigen::VectorXd thetalist(1);
Eigen::MatrixXd T = mymr::FK::FKinSpace(M, Slist, thetalist);
```

### IK

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

### InverseDynamics

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

### RobotControl

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
