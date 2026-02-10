# import numpy as np
# from Tools import *
#  # Using the library associated with your book

# def IKNumericDLS(J_func, Tsb_func, thetalist_guess, T_goal, 
#                 e_omg=0.001, e_v=0.001, max_iter=100, lam=0.1):
#     """
#     Uses Damped Least Squares to find joint angles for a goal pose.
    
#     :param J_func: Function that returns the Jacobian at current theta
#     :param Tsb_func: Function that returns current T_sb (Forward Kinematics)
#     :param thetalist_guess: Initial starting angles
#     :param T_goal: The 4x4 goal transformation matrix
#     :param e_omg/e_v: Error tolerances for orientation and position
#     :param lam: The damping factor (lambda) for DLS
#     :return: (thetalist, success)
#     """
#     thetalist = np.array(thetalist_guess).copy()
    
#     for i in range(max_iter):
#         # 1. Compute current pose and the error twist
#         T_current = Tsb_func(thetalist)
#         # Compute the twist needed to move from current to goal (in Body or Space frame)
#         # For Body Jacobian, we use mr.TransInv(T_current) @ T_goal
#         V_b_matrix = MatrixLog6(np.dot(TransInv(T_current), T_goal))
#         V_error = se3ToVec(V_b_matrix)
        
#         # 2. Check if the error is small enough
#         if np.linalg.norm(V_error[:3]) < e_omg and np.linalg.norm(V_error[3:]) < e_v:
#             return thetalist, True
        
#         # 3. Calculate the Jacobian
#         J = J_func(thetalist)
        
#         # 4. Damped Least Squares Formula:
#         # Instead of np.linalg.pinv(J), we use the damped version:
#         # d_theta = J^T * inv(J*J^T + lam^2 * I) * V_error
#         JJt = np.dot(J, J.T)
#         damping = (lam**2) * np.eye(6)
#         inv_part = np.linalg.inv(JJt + damping)
#         d_theta = np.dot(J.T, np.dot(inv_part, V_error))
        
#         # 5. Update the joint angles
#         thetalist = thetalist + d_theta
        
#     return thetalist, False