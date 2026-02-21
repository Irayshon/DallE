from Modern_Robotics_Py.Tools import *
from Modern_Robotics_Py.InverseDynamics import *
from Modern_Robotics_Py.ForwardDynamics import *
import numpy as np

def InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist):
    """
    Calculates the joint forces/torques required to move the robot along a given trajectory.
    
    :param thetamat: N x n matrix of joint variables
    :param dthetamat: N x n matrix of joint velocities
    :param ddthetamat: N x n matrix of joint accelerations
    :param g: Gravity vector
    :param Ftipmat: N x 6 matrix of spatial forces applied by the end-effector
    :return: N x n matrix of joint forces/torques
    """
    # Transpose to work with columns (time steps) for easier iteration
    thetamat = np.array(thetamat).T
    dthetamat = np.array(dthetamat).T
    ddthetamat = np.array(ddthetamat).T
    Ftipmat = np.array(Ftipmat).T
    
    N = thetamat.shape[1]
    taumat = np.zeros_like(thetamat)
    
    for i in range(N):
        taumat[:, i] = InverseDynamics(thetamat[:, i], dthetamat[:, i], 
                                      ddthetamat[:, i], g, Ftipmat[:, i], 
                                      Mlist, Glist, Slist)
    
    return taumat.T


def ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes):
    """
    Simulates the motion of a robot given an open-loop history of joint torques.
    
    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint rates
    :param taumat: N x n matrix of joint torques
    :param dt: Time step between torque commands
    :param intRes: Integration resolution (number of Euler steps per dt)
    :return: (thetamat, dthetamat) resulting from the torques
    """
    taumat = np.array(taumat).T
    Ftipmat = np.array(Ftipmat).T
    
    N = taumat.shape[1]
    thetamat = np.zeros((len(thetalist), N))
    dthetamat = np.zeros((len(dthetalist), N))
    
    # Initialize with the starting state
    current_theta = np.array(thetalist).copy()
    current_dtheta = np.array(dthetalist).copy()
    thetamat[:, 0] = current_theta
    dthetamat[:, 0] = current_dtheta
    
    for i in range(N - 1):
        # We perform 'intRes' number of small steps for every one torque command
        # This increases the numerical stability of your UR5e simulation
        for j in range(intRes):
            ddtheta = ForwardDynamics(current_theta, current_dtheta, taumat[:, i], 
                                     g, Ftipmat[:, i], Mlist, Glist, Slist)
            
            current_theta, current_dtheta = EulerStep(current_theta, current_dtheta, 
                                                     ddtheta, 1.0 * dt / intRes)
            
        thetamat[:, i + 1] = current_theta
        dthetamat[:, i + 1] = current_dtheta
        
    return thetamat.T, dthetamat.T