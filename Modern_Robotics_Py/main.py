import numpy as np
import pytest
from IK_MPC import IK_MPC # Assuming your code is in mpc_ik.py

# --- Helper functions to match the C++ logic ---

def get_planar_two_link_params():
    """Returns M, Slist, and Blist for the 2-link planar arm used in the tests."""
    M = np.array([[1, 0, 0, 2],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    # Space list: 2 joints rotating around Z axis
    Slist = np.array([[0, 0], [0, 0], [1, 1], [0, 0], [0, -1], [0, 0]])
    
    # Body list
    Blist = np.array([[0, 0], [0, 0], [1, 1], [0, 0], [2, 1], [0, 0]])
    
    return M, Slist, Blist

def get_three_dof_params():
    """Returns M, Slist, and Blist for the 3-DOF example."""
    M = np.array([[-1, 0, 0, 0],
                  [ 0, 1, 0, 6],
                  [ 0, 0,-1, 2],
                  [ 0, 0, 0, 1]])
    
    Blist = np.array([[ 0, 0, 0],
                      [ 0, 0, 0],
                      [-1, 0, 1],
                      [ 2, 0, 0],
                      [ 0, 1, 0],
                      [ 0, 0, 0.1]])
    
    Slist = np.array([[ 0, 0, 0],
                      [ 0, 0, 0],
                      [ 1, 0, -1],
                      [ 4, 0, -6],
                      [ 0, 1, 0],
                      [ 0, 0, -0.1]])
    return M, Slist, Blist

# --- Test Cases ---

def test_planar_two_link_target():
    """Python version of TEST(IKTest, PlanarTwoLinkTarget)"""
    M, Slist, Blist = get_planar_two_link_params()
    
    theta1 = np.pi / 4.0
    theta2 = -np.pi / 3.0
    theta12 = theta1 + theta2
    c12, s12 = np.cos(theta12), np.sin(theta12)
    x = np.cos(theta1) + np.cos(theta12)
    y = np.sin(theta1) + np.sin(theta12)

    T_goal = np.array([[c12, -s12, 0, x],
                       [s12,  c12, 0, y],
                       [  0,    0, 1, 0],
                       [  0,    0, 0, 1]])

    q0 = np.array([0.7, -1.0])
    Q = np.eye(6)
    R = np.eye(2) * 0.01

    # Test Body Frame
    q_res, debug = IK_MPC(q0, T_goal, N=5, Q=Q, R=R, M=M, screw_list=Blist, 
                          frame='body', dt=0.1, return_debug=True)
    
    assert debug['success'], f"IK failed to converge: {debug['err_v_norm']}"
    assert debug['err_v_norm'] < 1e-3

def test_mpc_joint_limits():
    """Python version of TEST(MPCIKTest, MPCThreeDOFJointLimits)"""
    M, _, Blist = get_three_dof_params()
    
    # Target pose (approximate goal)
    T_goal = np.array([[ 0, 1, 0, -5],
                       [ 1, 0, 0,  4],
                       [ 0, 0, -1, 1.6858],
                       [ 0, 0, 0,  1]])

    q0 = np.array([0.0, 0.0, 0.0])
    Q = np.eye(6)
    R = np.eye(3) * 0.1
    
    # Constraints
    qLim = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    uLim = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])

    q_res, debug = IK_MPC(q0, T_goal, N=5, Q=Q, R=R, M=M, screw_list=Blist, 
                          frame='body', qLim=qLim, uLim=uLim, dt=0.05, 
                          max_iters=300, return_debug=True)

    # Check that joint limits are respected
    for i in range(len(q_res)):
        assert q_res[i] >= qLim[i, 0] - 1e-6
        assert q_res[i] <= qLim[i, 1] + 1e-6

def test_mpc_non_convergence():
    """Python version of TEST(MPCIKTest, MPCNonConvergence)"""
    M, _, Blist = get_planar_two_link_params()
    
    # Target way out of reach
    T_far = np.array([[1, 0, 0, 10],
                      [0, 1, 0, 10],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    q0 = np.array([0.0, 0.0])
    Q = np.eye(6)
    R = np.eye(2)

    q_res, debug = IK_MPC(q0, T_far, N=5, Q=Q, R=R, M=M, screw_list=Blist, 
                          frame='body', max_iters=20, return_debug=True)

    assert not debug['success']
    assert debug['err_v_norm'] > 1.0

if __name__ == "__main__":
    # Run tests manually if not using pytest
    print("Running Planar Test...")
    test_planar_two_link_target()
    print("Planar Test Passed!")
    
    print("Running Joint Limits Test...")
    test_mpc_joint_limits()
    print("Joint Limits Test Passed!")
    
    print("Running Non-Convergence Test...")
    test_mpc_non_convergence()
    print("Non-Convergence Test Passed!")