
from Modern_Robotics_Py.Jacobian import *
from Modern_Robotics_Py.Tools import *
import numpy as np
from Modern_Robotics_Py.InverseDynamics import *

def ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist):
    """Computes joint accelerations given the current state and applied torques.
    :param thetalist: A list of joint variables
    :param dthetalist: A list of joint rates
    :param taulist: An n-vector of joint forces/torques
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The resulting joint accelerations(ddthetalist) as an n-vector
    """
    
    # 1. Calculate the components
    M = MassMatrix(thetalist, Mlist, Glist, Slist)
    c = VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
    grav = GravityForces(thetalist, g, Mlist, Glist, Slist)
    Ef = EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)

    # 2. Solve for acceleration: ddtheta = M^-1 * (tau - c - g - Ef)
    ddthetalist = np.linalg.solve(M, np.array(taulist) - c - grav -Ef)
    return ddthetalist