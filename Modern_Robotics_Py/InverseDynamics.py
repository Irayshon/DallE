import numpy as np
from Modern_Robotics_Py.Tools import *

def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    """
    Computes inverse dynamics using the Recursive Newton-Euler Algorithm.
    
    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: 3-vector gravity (e.g., [0, 0, -9.81])
    :param Ftip: 6-vector spatial force applied at the end-effector
    :param Mlist: List of link home frames {i} relative to {i-1}
    :param Glist: Spatial inertia matrices for each link
    :param Slist: Screw axes in the space frame
    :return: n-vector of joint torques
    """

    n = len(thetalist)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n + 1)
    Vi = np.zeros((6, n + 1))
    Vdi = np.zeros((6, n + 1))

    if g is None:
        g = [0, 0, 9.81]
    


    Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
    
    # Pre-calculate the transform from the last link to the end-effector
    AdTi[n] =    Adjoint(   TransInv(Mlist[n]))

    Fi = np.array(Ftip).copy()

    taulist = np.zeros(n)
    
    # --- Forward Pass ---
    for i in range(n):
        Mi = np.dot(Mi, Mlist[i])

        Ai[:, i] = np.dot(   Adjoint(   TransInv(Mi)), np.array(Slist)[:, i])
        
        AdTi[i] =    Adjoint(np.dot(
               MatrixExp6(   VecTose3(Ai[:, i] * -thetalist[i])),
               TransInv(Mlist[i])
        ))
        
        # Update link twist
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dthetalist[i]
        
        # Update link acceleration
        Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i]) + \
                       Ai[:, i] * ddthetalist[i] + \
                       np.dot(   ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]
    
    
    # --- Backward Pass ---
    for i in range(n - 1, -1, -1):
        # Sum forces
        Fi = np.dot(np.array(AdTi[i + 1]).T, Fi) + \
             np.dot(np.array(Glist[i]), Vdi[:, i + 1]) - \
             np.dot(np.array(   ad(Vi[:, i + 1])).T, 
                    np.dot(np.array(Glist[i]), Vi[:, i + 1]))
        
        # Project link wrench onto screw axis
        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])

    return taulist

# def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, \
#                     Glist, Slist):
#     """Computes inverse dynamics in the space frame for an open chain robot

#     :param thetalist: n-vector of joint variables
#     :param dthetalist: n-vector of joint rates
#     :param ddthetalist: n-vector of joint accelerations
#     :param g: Gravity vector g
#     :param Ftip: Spatial force applied by the end-effector expressed in frame
#                  {n+1}
#     :param Mlist: List of link frames {i} relative to {i-1} at the home
#                   position
#     :param Glist: Spatial inertia matrices Gi of the links
#     :param Slist: Screw axes Si of the joints in a space frame, in the format
#                   of a matrix with axes as the columns
#     :return: The n-vector of required joint forces/torques
#     This function uses forward-backward Newton-Euler iterations to solve the
#     equation:
#     taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) \
#               + g(thetalist) + Jtr(thetalist)Ftip

#     Example Input (3 Link Robot):
#         thetalist = np.array([0.1, 0.1, 0.1])
#         dthetalist = np.array([0.1, 0.2, 0.3])
#         ddthetalist = np.array([2, 1.5, 1])
#         g = np.array([0, 0, -9.8])
#         Ftip = np.array([1, 1, 1, 1, 1, 1])
#         M01 = np.array([[1, 0, 0,        0],
#                         [0, 1, 0,        0],
#                         [0, 0, 1, 0.089159],
#                         [0, 0, 0,        1]])
#         M12 = np.array([[ 0, 0, 1,    0.28],
#                         [ 0, 1, 0, 0.13585],
#                         [-1, 0, 0,       0],
#                         [ 0, 0, 0,       1]])
#         M23 = np.array([[1, 0, 0,       0],
#                         [0, 1, 0, -0.1197],
#                         [0, 0, 1,   0.395],
#                         [0, 0, 0,       1]])
#         M34 = np.array([[1, 0, 0,       0],
#                         [0, 1, 0,       0],
#                         [0, 0, 1, 0.14225],
#                         [0, 0, 0,       1]])
#         G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
#         G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
#         G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
#         Glist = np.array([G1, G2, G3])
#         Mlist = np.array([M01, M12, M23, M34])
#         Slist = np.array([[1, 0, 1,      0, 1,     0],
#                           [0, 1, 0, -0.089, 0,     0],
#                           [0, 1, 0, -0.089, 0, 0.425]]).T
#     Output:
#         np.array([74.69616155, -33.06766016, -3.23057314])
#     """
#     n = len(thetalist)
#     Mi = np.eye(4)
#     Ai = np.zeros((6, n))
#     AdTi = [[None]] * (n + 1)
#     Vi = np.zeros((6, n + 1))
#     Vdi = np.zeros((6, n + 1))
#     Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
#     AdTi[n] = Adjoint(TransInv(Mlist[n]))
#     Fi = np.array(Ftip).copy()
#     taulist = np.zeros(n)
#     for i in range(n):
#         Mi = np.dot(Mi,Mlist[i])
#         Ai[:, i] = np.dot(Adjoint(TransInv(Mi)), np.array(Slist)[:, i])
#         AdTi[i] = Adjoint(np.dot(MatrixExp6(VecTose3(Ai[:, i] * \
#                                             -thetalist[i])), \
#                                  TransInv(Mlist[i])))
#         Vi[:, i + 1] = np.dot(AdTi[i], Vi[:,i]) + Ai[:, i] * dthetalist[i]
#         Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i]) \
#                        + Ai[:, i] * ddthetalist[i] \
#                        + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]
#     for i in range (n - 1, -1, -1):
#         Fi = np.dot(np.array(AdTi[i + 1]).T, Fi) \
#              + np.dot(np.array(Glist[i]), Vdi[:, i + 1]) \
#              - np.dot(np.array(ad(Vi[:, i + 1])).T, \
#                       np.dot(np.array(Glist[i]), Vi[:, i + 1]))
#         taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])
#     return taulist

def MassMatrix(thetalist, Mlist, Glist, Slist):
    """
    Computes the mass matrix M(theta) for an open chain robot.
    
    :param thetalist: n-vector of joint variables
    :param Mlist: List of link home frames
    :param Glist: Spatial inertia matrices for each link
    :param Slist: Screw axes in the space frame
    :return: The n x n numerical mass matrix
    """

    n = len(thetalist)
    M = np.zeros((n, n))
    
    # We call InverseDynamics n times to build the matrix column by column
    for i in range(n):
        # Create a unit acceleration vector for the i-th joint
        ddthetalist = [0] * n
        ddthetalist[i] = 1
        
        M[:, i] = InverseDynamics(
            thetalist, 
            [0] * n, 
            ddthetalist, 
            [0, 0, 0], 
            [0, 0, 0, 0, 0, 0], 
            Mlist, 
            Glist, 
            Slist
        )

    return M

def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    """
    Computes the Coriolis and centripetal terms c(theta, dtheta).
    :param thetalist: A list of joint variables,
    :param dthetalist: A list of joint rates,
    :param Mlist: List of link frames i relative to i-1 at the home position,
    :param Glist: Spatial inertia matrices Gi of the links,
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns.
    :return: The vector c(thetalist,dthetalist) of Coriolis and centripetal
             terms for a given thetalist and dthetalist.
    This function calls InverseDynamics with g = 0, Ftip = 0, and
    ddthetalist = 0.
    """
    n = len(thetalist)
    # Call InverseDynamics with zero acceleration and zero gravity
    Cor = InverseDynamics(thetalist, dthetalist, [0] * n, [0, 0, 0], 
                            [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)
    return Cor


def GravityForces(thetalist, g, Mlist, Glist, Slist):
    """
    Computes the joint torques required to overcome gravity.
    :param thetalist: A list of joint variables
    :param g: 3-vector for gravitational acceleration
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return grav: The joint forces/torques required to overcome gravity at
                  thetalist
    This function calls InverseDynamics with Ftip = 0, dthetalist = 0, and
    ddthetalist = 0.
    """
    n = len(thetalist)
    # Call InverseDynamics with zero velocity and zero acceleration
    Gf = InverseDynamics(thetalist, [0] * n, [0] * n, g, 
                            [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)
    return Gf

def EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist):
    """Computes the joint forces/torques an open chain robot requires only to
    create the end-effector force Ftip

    :param thetalist: A list of joint variables
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The joint forces and torques required only to create the
             end-effector force Ftip
    This function calls InverseDynamics with g = 0, dthetalist = 0, and
    ddthetalist = 0.
    """
    n = len(thetalist)
    Ef = InverseDynamics(thetalist, [0] * n, [0] * n, [0, 0, 0], Ftip, Mlist, Glist, Slist)
    return Ef