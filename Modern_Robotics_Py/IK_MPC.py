import numpy as np
from scipy.linalg import block_diag
from qpsolvers import solve_qp as qp_solver



from Modern_Robotics_Py.ForwardKinematics import FKin
from Modern_Robotics_Py.Jacobian import Jacobian
from Modern_Robotics_Py.Tools import MatrixLog6, TransInv, se3ToVec

def BatchMatrices(N, A, B, Q, R, Pf):
    '''
    Docstring for BatchMatrices
    
    param N: Prediction horizon
    param A: State transition matrix
    param B: Control input matrix
    In discrete time, q(k+1) = q(k) + q_dot(k)*dt can be represented as q(k+1) = A*q(k) + B*u(k) where
    A is generally eye(n), where n is number of joints
    B is generally dt*eye(n), and dt is the time step size
    param Q: State cost matrix
    param R: Control cost matrix
    param Pf: Terminal state cost matrix
    '''
    n = B.shape[0]
    m = B.shape[1]

    Gamma = np.zeros((N * n, N * m))
    for i in range(N):
        for j in range(N):
            if i >= j:
                Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B

    omg = np.vstack([np.linalg.matrix_power(A, i) for i in range(N)])

    Q_list = [Q] * (N - 1) + [Pf]
    Qbar = block_diag(*Q_list)
    Rbar = block_diag(*([R] * N))

    Kb = -np.linalg.inv(Gamma.T @ Qbar @ Gamma + Rbar) @ (Gamma.T @ Qbar @ omg)
    K0N = Kb[:m, :]

    return Gamma, omg, Qbar, Rbar, K0N

def IK_MPC(
    q0,
    T_goal,
    N,
    Q,
    R,
    *,
    M=None,
    screw_list=None,
    frame: str = 'body',
    Pf=None,
    uLim=None,
    qLim=None,
    dt: float = 0.1,
    max_iters: int = 200,
    tol_omg: float = 1e-3,
    tol_v: float = 1e-3,
    solver: str = 'cvxopt',
    return_debug: bool = False,
):
    """Inverse kinematics via receding-horizon MPC (QP).

    Parameters
    - q0: (n,) initial joint angles
    - T_goal: (4,4) desired end-effector pose in the same base frame as FKin
    - N: prediction horizon (positive int)
    - Q: (6,6) pose error cost (twist error)
    - R: (n,n) joint velocity cost
    - M: (4,4) home configuration of end-effector
    - screw_list: (6,n) screw axes (Blist for frame='body', Slist for frame='space')
    - frame: 'body' or 'space' (selects which Jacobian/screw axes are used)
    - Pf: (6,6) terminal pose error cost (defaults to Q)
    - uLim: (n,2) joint velocity limits [[min,max],...]
    - qLim: (n,2) joint position limits [[min,max],...]
    - dt: time step
    - max_iters: maximum MPC iterations
    - tol_omg/tol_v: orientation/position tolerances on twist error

    Returns
    - q: (n,) joint angles that (approximately) achieve T_goal
    """

    q = np.array(q0, dtype=float).copy().reshape(-1)
    if Pf is None:
        Pf = Q

    if M is None or screw_list is None:
        raise ValueError("IK_MPC requires M and screw_list to compute forward kinematics and Jacobian")

    frame = frame.lower()
    if frame not in ('body', 'space'):
        raise ValueError("frame must be 'body' or 'space'")

    n = q.shape[0]
    if Q.shape != (6, 6):
        raise ValueError("Q must be 6x6 (task-space twist error cost)")
    if R.shape != (n, n):
        raise ValueError("R must be nxn (joint velocity cost)")

    debug = {
        'iters': 0,
        'success': False,
        'err_omg_norm': None,
        'err_v_norm': None,
        'q_history': [q.copy()],
        'V_history': [],
    }

    for it in range(max_iters):
        # Current pose and error twist
        fk_frame = 'body' if frame == 'body' else 'space'
        T_curr = FKin(M, screw_list, q, frame=fk_frame)

        if frame == 'body':
            T_err = TransInv(T_curr) @ T_goal
        else:
            # Space-frame error (left-invariant): T_goal * inv(T_curr)
            T_err = T_goal @ TransInv(T_curr)

        V_err = se3ToVec(MatrixLog6(T_err)).astype(float).reshape(6)
        err_omg = float(np.linalg.norm(V_err[:3]))
        err_v = float(np.linalg.norm(V_err[3:]))
        debug['V_history'].append(V_err.copy())

        if err_omg < tol_omg and err_v < tol_v:
            debug['iters'] = it
            debug['success'] = True
            debug['err_omg_norm'] = err_omg
            debug['err_v_norm'] = err_v
            break

        # Linearized error dynamics: x_{k+1} = x_k + B u_k
        # For body error, a reasonable local model is x_{k+1} ≈ x_k - dt * J(q) u_k.
        J = Jacobian(screw_list, q, frame=frame)
        A = np.eye(6)
        B = -dt * J  # (6,n)

        Gamma, omg, Qbar, Rbar, _ = BatchMatrices(N, A, B, Q, R, Pf)

        H = 2.0 * (Gamma.T @ Qbar @ Gamma + Rbar)
        H = 0.5 * (H + H.T) + 1e-9 * np.eye(H.shape[0])
        f_base = 2.0 * (Gamma.T @ Qbar @ omg)  # (N*n, 6)
        f = f_base @ V_err

        # Bounds on u over the whole horizon
        lb = None
        ub = None
        if uLim is not None:
            uLim = np.asarray(uLim, dtype=float)
            if uLim.shape != (n, 2):
                raise ValueError("uLim must be shape (n,2)")
            ulb = uLim[:, 0]
            uub = uLim[:, 1]
            lb = np.tile(ulb, N)
            ub = np.tile(uub, N)

        # Joint limits as linear inequality constraints on cumulative sum of u
        G = None
        h = None
        if qLim is not None:
            qLim = np.asarray(qLim, dtype=float)
            if qLim.shape != (n, 2):
                raise ValueError("qLim must be shape (n,2)")
            qlb = qLim[:, 0]
            qub = qLim[:, 1]

            Aq = np.eye(n)
            Bq = dt * np.eye(n)
            # We only need Gamma_q and omg_q; Q/R don't matter here.
            Gamma_q, omg_q, _, _, _ = BatchMatrices(N, Aq, Bq, np.eye(n), np.eye(n), np.eye(n))
            q_offset = (omg_q @ q).reshape(N * n)

            qlb_vec = np.tile(qlb, N)
            qub_vec = np.tile(qub, N)

            G = np.vstack([Gamma_q, -Gamma_q])
            h = np.hstack([qub_vec - q_offset, -(qlb_vec - q_offset)])

        u_seq = qp_solver(H, f, G=G, h=h, lb=lb, ub=ub, solver=solver)
        
        if u_seq is None:
            debug['iters'] = it
            debug['success'] = False
            debug['err_omg_norm'] = err_omg
            debug['err_v_norm'] = err_v
            break

        u0 = np.array(u_seq[:n], dtype=float)
        q = q + dt * u0
        debug['q_history'].append(q.copy())

    else:
        debug['iters'] = max_iters
        debug['success'] = False

    if debug['err_omg_norm'] is None or debug['err_v_norm'] is None:
        # Fill with last known error if we exited via break
        debug['err_omg_norm'] = float(np.linalg.norm(debug['V_history'][-1][:3])) if debug['V_history'] else None
        debug['err_v_norm'] = float(np.linalg.norm(debug['V_history'][-1][3:])) if debug['V_history'] else None

    return (q, debug) if return_debug else q