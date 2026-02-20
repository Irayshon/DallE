#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robot_commander_cpp {

using FKFunction = std::function<Eigen::Matrix4d(
    const Eigen::Matrix4d&, const Eigen::MatrixXd&, const Eigen::VectorXd&, const std::string&)>;
using JacobianFunction = std::function<Eigen::MatrixXd(
    const Eigen::MatrixXd&, const Eigen::VectorXd&, const std::string&)>;

struct IKMPCOptions {
  int horizon = 10;
  double dt = 0.1;
  int max_iters = 200;
  double tol_omg = 1e-3;
  double tol_v = 1e-3;
  std::string frame = "body";

  Eigen::VectorXd u_min;
  Eigen::VectorXd u_max;
  Eigen::VectorXd q_min;
  Eigen::VectorXd q_max;

  double q_limit_penalty = 50.0;
  int qp_inner_iters = 120;
  double qp_step = 0.05;
};

struct IKMPCDebug {
  int iters = 0;
  bool success = false;
  double err_omg_norm = std::numeric_limits<double>::quiet_NaN();
  double err_v_norm = std::numeric_limits<double>::quiet_NaN();
  std::vector<Eigen::VectorXd> q_history;
  std::vector<Eigen::VectorXd> v_history;
};

namespace {

constexpr double kEps = 1e-9;

bool IsSquare(const Eigen::MatrixXd& m) { return m.rows() == m.cols(); }

Eigen::Matrix3d VecToSo3(const Eigen::Vector3d& w) {
  Eigen::Matrix3d so3;
  so3 << 0.0, -w.z(), w.y(),
      w.z(), 0.0, -w.x(),
      -w.y(), w.x(), 0.0;
  return so3;
}

Eigen::Vector3d So3ToVec(const Eigen::Matrix3d& so3) {
  return Eigen::Vector3d(so3(2, 1), so3(0, 2), so3(1, 0));
}

Eigen::Matrix4d Se3FromVec(const Eigen::Matrix<double, 6, 1>& V) {
  Eigen::Matrix4d se3 = Eigen::Matrix4d::Zero();
  se3.block<3, 3>(0, 0) = VecToSo3(V.head<3>());
  se3.block<3, 1>(0, 3) = V.tail<3>();
  return se3;
}

Eigen::Matrix<double, 6, 1> Se3ToVec(const Eigen::Matrix4d& se3) {
  Eigen::Matrix<double, 6, 1> V;
  V.head<3>() = So3ToVec(se3.block<3, 3>(0, 0));
  V.tail<3>() = se3.block<3, 1>(0, 3);
  return V;
}

Eigen::Matrix4d TransInv(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d p = T.block<3, 1>(0, 3);
  Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
  Ti.block<3, 3>(0, 0) = R.transpose();
  Ti.block<3, 1>(0, 3) = -R.transpose() * p;
  return Ti;
}

Eigen::Matrix3d MatrixLog3(const Eigen::Matrix3d& R) {
  const double trace = R.trace();
  const double acos_input = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);

  if (acos_input >= 1.0 - 1e-12) {
    return Eigen::Matrix3d::Zero();
  }

  if (acos_input <= -1.0 + 1e-12) {
    Eigen::Vector3d omg;
    if (std::abs(1.0 + R(2, 2)) > 1e-12) {
      omg = (1.0 / std::sqrt(2.0 * (1.0 + R(2, 2)))) *
            Eigen::Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2));
    } else if (std::abs(1.0 + R(1, 1)) > 1e-12) {
      omg = (1.0 / std::sqrt(2.0 * (1.0 + R(1, 1)))) *
            Eigen::Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1));
    } else {
      omg = (1.0 / std::sqrt(2.0 * (1.0 + R(0, 0)))) *
            Eigen::Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0));
    }
    return VecToSo3(M_PI * omg);
  }

  const double theta = std::acos(acos_input);
  return (theta / (2.0 * std::sin(theta))) * (R - R.transpose());
}

Eigen::Matrix4d MatrixLog6(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d p = T.block<3, 1>(0, 3);
  const Eigen::Matrix3d omgmat = MatrixLog3(R);

  Eigen::Matrix4d se3 = Eigen::Matrix4d::Zero();
  se3.block<3, 3>(0, 0) = omgmat;

  if (omgmat.norm() < 1e-12) {
    se3.block<3, 1>(0, 3) = p;
    return se3;
  }

  const double theta = std::acos(std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0));
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d V_inv =
      I - (0.5 * omgmat) +
      ((1.0 / theta) - (0.5 / std::tan(0.5 * theta))) * (omgmat * omgmat) / theta;

  se3.block<3, 1>(0, 3) = V_inv * p;
  return se3;
}

struct BatchMatricesOut {
  Eigen::MatrixXd gamma;
  Eigen::MatrixXd omega;
  Eigen::MatrixXd q_bar;
  Eigen::MatrixXd r_bar;
};

BatchMatricesOut BatchMatrices(
    int N,
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& Pf) {
  const int n = static_cast<int>(B.rows());
  const int m = static_cast<int>(B.cols());

  Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(N * n, N * m);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i >= j) {
        Eigen::MatrixXd A_pow = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        for (int k = 0; k < (i - j); ++k) {
          A_pow = A_pow * A;
        }
        gamma.block(i * n, j * m, n, m) = A_pow * B;
      }
    }
  }

  Eigen::MatrixXd omega(N * n, n);
  for (int i = 0; i < N; ++i) {
    Eigen::MatrixXd A_pow = Eigen::MatrixXd::Identity(A.rows(), A.cols());
    for (int k = 0; k < i; ++k) {
      A_pow = A_pow * A;
    }
    omega.block(i * n, 0, n, n) = A_pow;
  }

  Eigen::MatrixXd q_bar = Eigen::MatrixXd::Zero(N * Q.rows(), N * Q.cols());
  for (int i = 0; i < N - 1; ++i) {
    q_bar.block(i * Q.rows(), i * Q.cols(), Q.rows(), Q.cols()) = Q;
  }
  q_bar.block((N - 1) * Q.rows(), (N - 1) * Q.cols(), Pf.rows(), Pf.cols()) = Pf;

  Eigen::MatrixXd r_bar = Eigen::MatrixXd::Zero(N * R.rows(), N * R.cols());
  for (int i = 0; i < N; ++i) {
    r_bar.block(i * R.rows(), i * R.cols(), R.rows(), R.cols()) = R;
  }

  return {gamma, omega, q_bar, r_bar};
}

void ProjectBox(Eigen::VectorXd* u, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
  for (int i = 0; i < u->size(); ++i) {
    (*u)(i) = std::clamp((*u)(i), lb(i), ub(i));
  }
}

Eigen::VectorXd SolveQPApprox(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& f,
    const IKMPCOptions& opt,
    const Eigen::MatrixXd& gamma_q,
    const Eigen::VectorXd& q_off,
    const Eigen::VectorXd& q_lb,
    const Eigen::VectorXd& q_ub) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
  Eigen::VectorXd u;
  if (ldlt.info() == Eigen::Success) {
    u = ldlt.solve(-f);
  } else {
    u = Eigen::VectorXd::Zero(f.size());
  }

  const bool has_u_box =
      (opt.u_min.size() == opt.u_max.size()) && (opt.u_min.size() > 0) && (opt.u_min.size() <= u.size());
  if (has_u_box) {
    Eigen::VectorXd lb(u.size());
    Eigen::VectorXd ub(u.size());
    const int n = opt.u_min.size();
    for (int i = 0; i < (u.size() / n); ++i) {
      lb.segment(i * n, n) = opt.u_min;
      ub.segment(i * n, n) = opt.u_max;
    }
    ProjectBox(&u, lb, ub);
  }

  const bool has_q_box = (q_lb.size() == q_ub.size()) && (q_lb.size() > 0);
  if (!has_q_box && !has_u_box) {
    return u;
  }

  for (int it = 0; it < opt.qp_inner_iters; ++it) {
    Eigen::VectorXd grad = H * u + f;

    if (has_q_box) {
      const Eigen::VectorXd q_pred = gamma_q * u + q_off;
      Eigen::VectorXd violation = Eigen::VectorXd::Zero(q_pred.size());
      for (int i = 0; i < q_pred.size(); ++i) {
        if (q_pred(i) < q_lb(i)) {
          violation(i) = q_pred(i) - q_lb(i);
        } else if (q_pred(i) > q_ub(i)) {
          violation(i) = q_pred(i) - q_ub(i);
        }
      }
      grad += 2.0 * opt.q_limit_penalty * (gamma_q.transpose() * violation);
    }

    u = u - opt.qp_step * grad;

    if (has_u_box) {
      Eigen::VectorXd lb(u.size());
      Eigen::VectorXd ub(u.size());
      const int n = opt.u_min.size();
      for (int i = 0; i < (u.size() / n); ++i) {
        lb.segment(i * n, n) = opt.u_min;
        ub.segment(i * n, n) = opt.u_max;
      }
      ProjectBox(&u, lb, ub);
    }
  }

  return u;
}

void ValidateInputs(
    const Eigen::VectorXd& q0,
    const Eigen::Matrix4d& T_goal,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& M,
    const Eigen::MatrixXd& screw_list,
    const IKMPCOptions& opt) {
  (void)T_goal;

  if (q0.size() == 0) {
    throw std::invalid_argument("q0 must be non-empty");
  }
  if (opt.horizon <= 0) {
    throw std::invalid_argument("horizon must be > 0");
  }
  if (opt.dt <= 0.0) {
    throw std::invalid_argument("dt must be > 0");
  }
  if (Q.rows() != 6 || Q.cols() != 6) {
    throw std::invalid_argument("Q must be 6x6");
  }
  if (!IsSquare(R) || R.rows() != q0.size()) {
    throw std::invalid_argument("R must be nxn where n=q0.size()");
  }
  if (M.rows() != 4 || M.cols() != 4) {
    throw std::invalid_argument("M must be 4x4");
  }
  if (screw_list.rows() != 6 || screw_list.cols() != q0.size()) {
    throw std::invalid_argument("screw_list must be 6xn and n=q0.size()");
  }
  if (opt.frame != "body" && opt.frame != "space") {
    throw std::invalid_argument("frame must be 'body' or 'space'");
  }

  if ((opt.u_min.size() > 0 || opt.u_max.size() > 0) &&
      (opt.u_min.size() != q0.size() || opt.u_max.size() != q0.size())) {
    throw std::invalid_argument("u_min/u_max must be size n if provided");
  }
  if ((opt.q_min.size() > 0 || opt.q_max.size() > 0) &&
      (opt.q_min.size() != q0.size() || opt.q_max.size() != q0.size())) {
    throw std::invalid_argument("q_min/q_max must be size n if provided");
  }
}

}

std::pair<Eigen::VectorXd, IKMPCDebug> IKMPC(
    const Eigen::VectorXd& q0,
    const Eigen::Matrix4d& T_goal,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& M,
    const Eigen::MatrixXd& screw_list,
    const FKFunction& fkin,
    const JacobianFunction& jacobian,
    const IKMPCOptions& opt,
    const Eigen::MatrixXd* Pf_override) {
  ValidateInputs(q0, T_goal, Q, R, M, screw_list, opt);

  const int n = q0.size();
  const int N = opt.horizon;
  const Eigen::MatrixXd Pf = (Pf_override == nullptr) ? Q : *Pf_override;
  if (Pf.rows() != 6 || Pf.cols() != 6) {
    throw std::invalid_argument("Pf must be 6x6");
  }

  Eigen::VectorXd q = q0;
  IKMPCDebug debug;
  debug.q_history.push_back(q);

  for (int it = 0; it < opt.max_iters; ++it) {
    const Eigen::Matrix4d T_curr = fkin(M, screw_list, q, opt.frame);

    Eigen::Matrix4d T_err;
    if (opt.frame == "body") {
      T_err = TransInv(T_curr) * T_goal;
    } else {
      T_err = T_goal * TransInv(T_curr);
    }

    const Eigen::Matrix<double, 6, 1> V_err = Se3ToVec(MatrixLog6(T_err));
    const double err_omg = V_err.head<3>().norm();
    const double err_v = V_err.tail<3>().norm();

    debug.v_history.push_back(V_err);
    debug.err_omg_norm = err_omg;
    debug.err_v_norm = err_v;

    if (err_omg < opt.tol_omg && err_v < opt.tol_v) {
      debug.iters = it;
      debug.success = true;
      return {q, debug};
    }

    const Eigen::MatrixXd J = jacobian(screw_list, q, opt.frame);
    if (J.rows() != 6 || J.cols() != n) {
      throw std::runtime_error("jacobian callback must return 6xn matrix");
    }

    const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(6, 6);
    const Eigen::MatrixXd B = -opt.dt * J;
    const BatchMatricesOut bm = BatchMatrices(N, A, B, Q, R, Pf);

    Eigen::MatrixXd H = 2.0 * (bm.gamma.transpose() * bm.q_bar * bm.gamma + bm.r_bar);
    H = 0.5 * (H + H.transpose()) + kEps * Eigen::MatrixXd::Identity(H.rows(), H.cols());
    const Eigen::MatrixXd f_base = 2.0 * (bm.gamma.transpose() * bm.q_bar * bm.omega);
    const Eigen::VectorXd f = f_base * V_err;

    const Eigen::MatrixXd Aq = Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd Bq = opt.dt * Eigen::MatrixXd::Identity(n, n);
    const BatchMatricesOut bq = BatchMatrices(
        N,
        Aq,
        Bq,
        Eigen::MatrixXd::Identity(n, n),
        Eigen::MatrixXd::Identity(n, n),
        Eigen::MatrixXd::Identity(n, n));
    const Eigen::VectorXd q_off = bq.omega * q;

    Eigen::VectorXd qlb = Eigen::VectorXd::Constant(N * n, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd qub = Eigen::VectorXd::Constant(N * n, std::numeric_limits<double>::infinity());
    if (opt.q_min.size() == n && opt.q_max.size() == n) {
      for (int k = 0; k < N; ++k) {
        qlb.segment(k * n, n) = opt.q_min;
        qub.segment(k * n, n) = opt.q_max;
      }
    }

    const Eigen::VectorXd u_seq = SolveQPApprox(H, f, opt, bq.gamma, q_off, qlb, qub);
    if (u_seq.size() != N * n) {
      debug.iters = it;
      debug.success = false;
      return {q, debug};
    }

    const Eigen::VectorXd u0 = u_seq.head(n);
    q = q + opt.dt * u0;
    debug.q_history.push_back(q);
  }

  debug.iters = opt.max_iters;
  debug.success = false;
  return {q, debug};
}

}
