#include "WallE/ik.h"

#include <algorithm>

#include <OsqpEigen/OsqpEigen.h>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

#include "WallE/fk.h"
#include "WallE/tools.h"

namespace {
constexpr double kLargeBound = 1e10;
constexpr double kFiniteDifferenceEpsilon = 1e-6;

Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& J) {
  if (J.size() == 0) {
    return Eigen::MatrixXd::Zero(J.cols(), J.rows());
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singular_values = svd.singularValues();
  double tolerance = 1e-6 * std::max(J.rows(), J.cols()) *
                     singular_values.array().abs()(0);
  Eigen::VectorXd singular_inverse = singular_values.unaryExpr(
      [tolerance](double sigma) { return sigma > tolerance ? 1.0 / sigma : 0.0; });
  return svd.matrixV() * singular_inverse.asDiagonal() *
         svd.matrixU().transpose();
}

Eigen::VectorXd ComputeTaskError(const Eigen::MatrixXd& T_current,
                                 const Eigen::Matrix4d& T_target) {
  return WallE::Tools::se3ToVec(
      WallE::Tools::MatrixLog6(WallE::Tools::TransInv(T_current) * T_target));
}

double ComputeManipulability(const Eigen::MatrixXd& J) {
  if (J.rows() == 0 || J.cols() == 0) {
    return 0.0;
  }
  Eigen::MatrixXd JJt = J * J.transpose();
  double determinant = JJt.determinant();
  return determinant > 0.0 ? std::sqrt(determinant) : 0.0;
}

using FKFunc = std::function<Eigen::MatrixXd(
    const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&)>;
using JacFunc = std::function<Eigen::MatrixXd(
    const Eigen::MatrixXd&, const Eigen::VectorXd&)>;
using ErrorTransformFunc = std::function<Eigen::VectorXd(
    const Eigen::MatrixXd& T_current, const Eigen::VectorXd& Vb)>;

std::vector<Eigen::Vector3d> ComputeLinkPositions(
    const Eigen::MatrixXd& M,
    const Eigen::MatrixXd& screw_axes,
    const Eigen::VectorXd& thetalist,
    const FKFunc& fk_func) {
  int n = static_cast<int>(thetalist.size());
  std::vector<Eigen::Vector3d> positions(n + 1, Eigen::Vector3d::Zero());
  for (int i = 1; i <= n; ++i) {
    Eigen::MatrixXd sub_axes = screw_axes.leftCols(i);
    Eigen::VectorXd sub_theta = thetalist.head(i);
    Eigen::MatrixXd T_i = fk_func(M, sub_axes, sub_theta);
    positions[i] = T_i.block<3, 1>(0, 3);
  }
  return positions;
}

Eigen::VectorXd NumericalGradient(
    const Eigen::VectorXd& theta,
    const std::function<double(const Eigen::VectorXd&)>& scalar_function) {
  int n = static_cast<int>(theta.size());
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);
  if (n == 0) {
    return gradient;
  }

  double baseline = scalar_function(theta);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd perturbed = theta;
    perturbed(i) += kFiniteDifferenceEpsilon;
    gradient(i) = (scalar_function(perturbed) - baseline) /
                  kFiniteDifferenceEpsilon;
  }
  return gradient;
}

struct PredictionStep {
  Eigen::VectorXd theta;
  Eigen::VectorXd dtheta;
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd task_error;
  Eigen::Vector3d ee_position = Eigen::Vector3d::Zero();
};

struct QPData {
  Eigen::SparseMatrix<double> H;
  Eigen::VectorXd g;
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd lb;
  Eigen::VectorXd ub;
  int num_vars = 0;
  int num_constraints = 0;
};

bool HasJointBounds(const Eigen::VectorXd& lower,
                    const Eigen::VectorXd& upper,
                    int n) {
  return lower.size() == n && upper.size() == n;
}

bool HasWorkspaceBounds(const WallE::MPCIKConfig& config) {
  return (config.workspace_min.array() > -1e9).any() ||
         (config.workspace_max.array() < 1e9).any();
}

QPData BuildMPCQP(
    const Eigen::MatrixXd& screw_axes,
    const Eigen::MatrixXd& M,
    const std::vector<PredictionStep>& predictions,
    const Eigen::VectorXd& theta_current,
    const Eigen::VectorXd& dtheta_current,
    double h,
    const WallE::MPCIKConfig& config,
    const FKFunc& fk_func,
    const JacFunc& jac_func) {
  QPData qp;

  int n = static_cast<int>(theta_current.size());
  int N = static_cast<int>(predictions.size());

  bool has_theta_bounds = HasJointBounds(config.theta_min, config.theta_max, n);
  bool has_velocity_bounds =
      HasJointBounds(config.dtheta_min, config.dtheta_max, n);
  bool has_acceleration_bounds =
      HasJointBounds(config.ddtheta_min, config.ddtheta_max, n);
  bool has_workspace_bounds = HasWorkspaceBounds(config);
  bool has_collision_bounds = !config.collision_pairs.empty();
  bool has_singularity_bounds = config.manipulability_threshold > 0.0 && n >= 6;

  qp.num_vars = N * n;
  qp.num_constraints = 3 * N * n;
  if (has_workspace_bounds) {
    qp.num_constraints += 3 * N;
  }
  if (has_collision_bounds) {
    qp.num_constraints += static_cast<int>(config.collision_pairs.size()) * N;
  }
  if (has_singularity_bounds) {
    qp.num_constraints += N;
  }

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6) * config.w_tracking;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n) * config.w_velocity;
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n) * config.w_acceleration;

  std::vector<Eigen::Triplet<double>> h_triplets;
  h_triplets.reserve(static_cast<size_t>(N) * n * (n + 1) / 2);
  qp.g = Eigen::VectorXd::Zero(qp.num_vars);

  for (int k = 0; k < N; ++k) {
    Eigen::MatrixXd H_k = predictions[k].jacobian.transpose() * Q *
                          predictions[k].jacobian * h * h +
                          R * h * h + S * h * h;
    H_k = 0.5 * (H_k + H_k.transpose());

    int offset = k * n;
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        if (std::abs(H_k(i, j)) > 1e-12) {
          h_triplets.emplace_back(offset + i, offset + j, H_k(i, j));
        }
      }
    }

    qp.g.segment(offset, n) =
        -predictions[k].jacobian.transpose() * Q * predictions[k].task_error * h +
        R * predictions[k].dtheta * h * h;
  }

  qp.H.resize(qp.num_vars, qp.num_vars);
  qp.H.setFromTriplets(h_triplets.begin(), h_triplets.end());
  qp.H = qp.H.triangularView<Eigen::Upper>();
  qp.H.makeCompressed();

  qp.lb = Eigen::VectorXd::Zero(qp.num_constraints);
  qp.ub = Eigen::VectorXd::Zero(qp.num_constraints);
  std::vector<Eigen::Triplet<double>> a_triplets;
  a_triplets.reserve(static_cast<size_t>(qp.num_constraints) *
                     std::max(1, n));

  int row = 0;
  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= k; ++j) {
        double coefficient = static_cast<double>(k - j + 1) * h;
        a_triplets.emplace_back(row + i, j * n + i, coefficient);
      }

      double theta_nominal =
          theta_current(i) + dtheta_current(i) * static_cast<double>(k + 1) * h;
      qp.lb(row + i) = has_theta_bounds ? config.theta_min(i) - theta_nominal
                                        : -kLargeBound;
      qp.ub(row + i) = has_theta_bounds ? config.theta_max(i) - theta_nominal
                                        : kLargeBound;
    }
    row += n;

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= k; ++j) {
        a_triplets.emplace_back(row + i, j * n + i, 1.0);
      }

      qp.lb(row + i) = has_velocity_bounds
                           ? config.dtheta_min(i) - dtheta_current(i)
                           : -kLargeBound;
      qp.ub(row + i) = has_velocity_bounds
                           ? config.dtheta_max(i) - dtheta_current(i)
                           : kLargeBound;
    }
    row += n;

    for (int i = 0; i < n; ++i) {
      a_triplets.emplace_back(row + i, k * n + i, 1.0 / h);
      qp.lb(row + i) = has_acceleration_bounds ? config.ddtheta_min(i)
                                               : -kLargeBound;
      qp.ub(row + i) = has_acceleration_bounds ? config.ddtheta_max(i)
                                               : kLargeBound;
    }
    row += n;
  }

  if (has_workspace_bounds) {
    for (int k = 0; k < N; ++k) {
      Eigen::MatrixXd linear_jacobian = predictions[k].jacobian.bottomRows(3);
      for (int axis = 0; axis < 3; ++axis) {
        for (int j = 0; j <= k; ++j) {
          double step_coefficient = static_cast<double>(k - j + 1) * h;
          for (int joint = 0; joint < n; ++joint) {
            double coefficient = linear_jacobian(axis, joint) * step_coefficient;
            if (std::abs(coefficient) > 1e-12) {
              a_triplets.emplace_back(row + axis, j * n + joint, coefficient);
            }
          }
        }

        qp.lb(row + axis) =
            config.workspace_min(axis) - predictions[k].ee_position(axis);
        qp.ub(row + axis) =
            config.workspace_max(axis) - predictions[k].ee_position(axis);
      }
      row += 3;
    }
  }

  if (has_collision_bounds) {
    for (int k = 0; k < N; ++k) {
      auto link_positions =
          ComputeLinkPositions(M, screw_axes, predictions[k].theta, fk_func);
      int max_index = static_cast<int>(link_positions.size()) - 1;

      for (size_t pair_index = 0; pair_index < config.collision_pairs.size();
           ++pair_index) {
        auto [link_i, link_j, min_distance] = config.collision_pairs[pair_index];
        int link_i_index = std::clamp(link_i, 0, max_index);
        int link_j_index = std::clamp(link_j, 0, max_index);

        auto distance_function =
            [&](const Eigen::VectorXd& theta_candidate) -> double {
          auto candidate_positions =
              ComputeLinkPositions(M, screw_axes, theta_candidate, fk_func);
          return (candidate_positions[link_i_index] -
                  candidate_positions[link_j_index])
              .norm();
        };

        double current_distance = distance_function(predictions[k].theta);
        Eigen::VectorXd distance_gradient =
            NumericalGradient(predictions[k].theta, distance_function);

        for (int j = 0; j <= k; ++j) {
          double step_coefficient = static_cast<double>(k - j + 1) * h;
          for (int joint = 0; joint < n; ++joint) {
            double coefficient = distance_gradient(joint) * step_coefficient;
            if (std::abs(coefficient) > 1e-12) {
              a_triplets.emplace_back(row, j * n + joint, coefficient);
            }
          }
        }

        qp.lb(row) = min_distance - current_distance;
        qp.ub(row) = kLargeBound;
        ++row;
      }
    }
  }

  if (has_singularity_bounds) {
    for (int k = 0; k < N; ++k) {
      auto manipulability_function =
          [&](const Eigen::VectorXd& theta_candidate) -> double {
        return ComputeManipulability(jac_func(screw_axes, theta_candidate));
      };

      double current_manipulability =
          manipulability_function(predictions[k].theta);
      Eigen::VectorXd manipulability_gradient =
          NumericalGradient(predictions[k].theta, manipulability_function);

      for (int j = 0; j <= k; ++j) {
        double step_coefficient = static_cast<double>(k - j + 1) * h;
        for (int joint = 0; joint < n; ++joint) {
          double coefficient = manipulability_gradient(joint) * step_coefficient;
          if (std::abs(coefficient) > 1e-12) {
            a_triplets.emplace_back(row, j * n + joint, coefficient);
          }
        }
      }

      qp.lb(row) = config.manipulability_threshold - current_manipulability;
      qp.ub(row) = kLargeBound;
      ++row;
    }
  }

  if (row != qp.num_constraints) {
    qp.num_constraints = row;
    qp.lb.conservativeResize(row);
    qp.ub.conservativeResize(row);
  }

  qp.A.resize(qp.num_constraints, qp.num_vars);
  qp.A.setFromTriplets(a_triplets.begin(), a_triplets.end());
  qp.A.makeCompressed();

  return qp;
}

WallE::MPCIKResult MPCIKSolve(const Eigen::MatrixXd& screw_axes,
                              const Eigen::MatrixXd& M,
                              const std::vector<Eigen::Matrix4d>& T_trajectory,
                              Eigen::VectorXd& thetalist,
                              const WallE::MPCIKConfig& config,
                              const FKFunc& fk_func,
                              const JacFunc& jac_func,
                              const ErrorTransformFunc& error_transform) {
  WallE::MPCIKResult result;
  int n = static_cast<int>(thetalist.size());

  result.thetalist = thetalist;
  result.dthetalist = Eigen::VectorXd::Zero(n);

  if (T_trajectory.empty()) {
    result.converged = true;
    return result;
  }

  int N = std::max(1, config.horizon);
  double h = config.dt > 0.0 ? config.dt : 1e-3;
  int max_iterations = std::max(1, config.max_iterations);

  Eigen::VectorXd theta = thetalist;
  Eigen::VectorXd dtheta = Eigen::VectorXd::Zero(n);
  int trajectory_index = 0;

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    int target_index =
        std::min(trajectory_index, static_cast<int>(T_trajectory.size()) - 1);
    Eigen::MatrixXd T_current = fk_func(M, screw_axes, theta);
    Eigen::VectorXd error = ComputeTaskError(T_current, T_trajectory[target_index]);
    double orientation_error = error.head(3).norm();
    double position_error = error.tail(3).norm();

    result.final_orientation_error = orientation_error;
    result.final_position_error = position_error;

    if (orientation_error <= config.eomg && position_error <= config.ev) {
      if (target_index == static_cast<int>(T_trajectory.size()) - 1) {
        result.converged = true;
        result.iterations = iteration;
        result.thetalist = theta;
        result.dthetalist = dtheta;
        thetalist = theta;
        return result;
      }
      ++trajectory_index;
      continue;
    }

    std::vector<PredictionStep> predictions(N);
    Eigen::VectorXd theta_prediction = theta;
    Eigen::VectorXd dtheta_prediction = dtheta;

    for (int k = 0; k < N; ++k) {
      int reference_index =
          std::min(trajectory_index + k,
                   static_cast<int>(T_trajectory.size()) - 1);
      Eigen::MatrixXd T_prediction = fk_func(M, screw_axes, theta_prediction);

      predictions[k].theta = theta_prediction;
      predictions[k].dtheta = dtheta_prediction;
      predictions[k].jacobian = jac_func(screw_axes, theta_prediction);
      Eigen::VectorXd body_error =
          ComputeTaskError(T_prediction, T_trajectory[reference_index]);
      predictions[k].task_error = error_transform(T_prediction, body_error);
      predictions[k].ee_position = T_prediction.block<3, 1>(0, 3);

      theta_prediction = theta_prediction + dtheta_prediction * h;
    }

    QPData qp = BuildMPCQP(screw_axes,
                           M,
                           predictions,
                           theta,
                           dtheta,
                           h,
                           config,
                           fk_func,
                           jac_func);

    if (qp.num_vars == 0 || qp.num_constraints == 0) {
      break;
    }

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    solver.data()->setNumberOfVariables(qp.num_vars);
    solver.data()->setNumberOfConstraints(qp.num_constraints);

    if (!solver.data()->setHessianMatrix(qp.H)) {
      break;
    }
    if (!solver.data()->setGradient(qp.g)) {
      break;
    }
    if (!solver.data()->setLinearConstraintsMatrix(qp.A)) {
      break;
    }
    if (!solver.data()->setLowerBound(qp.lb)) {
      break;
    }
    if (!solver.data()->setUpperBound(qp.ub)) {
      break;
    }

    if (!solver.initSolver()) {
      break;
    }

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      break;
    }

    Eigen::VectorXd solution = solver.getSolution();
    if (solution.size() < n) {
      break;
    }

    Eigen::VectorXd delta_dtheta_0 = solution.head(n);
    dtheta = delta_dtheta_0;
    theta += dtheta * h;

    result.iterations = iteration + 1;
  }

  int final_target_index =
      std::min(trajectory_index, static_cast<int>(T_trajectory.size()) - 1);
  Eigen::MatrixXd T_final = fk_func(M, screw_axes, theta);
  Eigen::VectorXd final_error =
      ComputeTaskError(T_final, T_trajectory[final_target_index]);
  result.final_orientation_error = final_error.head(3).norm();
  result.final_position_error = final_error.tail(3).norm();
  result.converged = false;
  result.thetalist = theta;
  result.dthetalist = dtheta;
  thetalist = theta;
  return result;
}
}  // namespace

namespace WallE {
bool IK::IKinBody(const Eigen::MatrixXd& Blist,
                  const Eigen::MatrixXd& M,
                  const Eigen::MatrixXd& T,
                  Eigen::VectorXd& thetalist,
                  double eomg,
                  double ev) {
  const int max_iterations = 20;
  int iteration = 0;
  Eigen::MatrixXd Tsb = FK::FKinBody(M, Blist, thetalist);
  Eigen::VectorXd Vb = Tools::se3ToVec(
      Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
  bool err = (Vb.head(3).norm() > eomg) || (Vb.tail(3).norm() > ev);

  while (err && iteration < max_iterations) {
    Eigen::MatrixXd Jb = Tools::JacobianBody(Blist, thetalist);
    thetalist = thetalist + PseudoInverse(Jb) * Vb;
    ++iteration;
    Tsb = FK::FKinBody(M, Blist, thetalist);
    Vb = Tools::se3ToVec(Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
    err = (Vb.head(3).norm() > eomg) || (Vb.tail(3).norm() > ev);
  }

  return !err;
}

bool IK::IKinSpace(const Eigen::MatrixXd& Slist,
                   const Eigen::MatrixXd& M,
                   const Eigen::MatrixXd& T,
                   Eigen::VectorXd& thetalist,
                   double eomg,
                   double ev) {
  const int max_iterations = 20;
  int iteration = 0;
  Eigen::MatrixXd Tsb = FK::FKinSpace(M, Slist, thetalist);
  Eigen::VectorXd Vb = Tools::se3ToVec(
      Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
  Eigen::VectorXd Vs = Tools::Adjoint(Tsb) * Vb;
  bool err = (Vs.head(3).norm() > eomg) || (Vs.tail(3).norm() > ev);

  while (err && iteration < max_iterations) {
    Eigen::MatrixXd Js = Tools::JacobianSpace(Slist, thetalist);
    thetalist = thetalist + PseudoInverse(Js) * Vs;
    ++iteration;
    Tsb = FK::FKinSpace(M, Slist, thetalist);
    Vb = Tools::se3ToVec(Tools::MatrixLog6(Tools::TransInv(Tsb) * T));
    Vs = Tools::Adjoint(Tsb) * Vb;
    err = (Vs.head(3).norm() > eomg) || (Vs.tail(3).norm() > ev);
  }

  return !err;
}

MPCIKResult IK::MPCIKinBody(const Eigen::MatrixXd& Blist,
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
  auto error_tf = [](const Eigen::MatrixXd& /*T_current*/,
                     const Eigen::VectorXd& Vb) {
    return Vb;
  };
  return MPCIKSolve(Blist, M, T_trajectory, thetalist, config, fk, jac,
                    error_tf);
}

MPCIKResult IK::MPCIKinSpace(const Eigen::MatrixXd& Slist,
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
  auto error_tf = [](const Eigen::MatrixXd& T_current,
                     const Eigen::VectorXd& Vb) -> Eigen::VectorXd {
    return Tools::Adjoint(T_current) * Vb;
  };
  return MPCIKSolve(Slist, M, T_trajectory, thetalist, config, fk, jac,
                    error_tf);
}
}  // namespace WallE
