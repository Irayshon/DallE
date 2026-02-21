import numpy as np

import numpy as np
import trimesh
import xml.etree.ElementTree as ET


'''__________________________________________________________________________________'''

def isZero(z):
    """Determines whether a scalar is small enough to be treated as zero

    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise

    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6


'''__________________________________________________________________________________'''


def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


'''__________________________________________________________________________________'''


def VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3

    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]


'''__________________________________________________________________________________'''

def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


'''__________________________________________________________________________________'''


def se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

'''__________________________________________________________________________________'''


def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (expc3 / np.linalg.norm(expc3), np.linalg.norm(expc3))


'''__________________________________________________________________________________'''

def GetScrew(w, L):
    w = np.array(w)
    L = np.array(L)
    v = np.cross(L, w)
    return np.concatenate((w, v))

'''__________________________________________________________________________________'''


def ConstructScrewlist(screw, radius):
    """
    Constructs a 6xn matrix of body screw axes.
    
    :param w_vals: List of 3-vectors (unit axes)
    :param L_vals: List of 3-vectors (points on axes)
    :return: A 6xn numpy array where each column is a screw axis [w; v]

    Example Input:
       w_vals = [
            [0, 0, 1], [1, 0, 0], [1, 0, 0], 
            [0, 0, 1], [1, 0, 0], [0, 0, 1]
        ]
        L_vals = [
            [0, 0, -16], [2, 0, -12], [0, 0, -8], 
            [0, 0, -4], [0, 0, -3], [0, 0, -1]
        ]

    Output:
            [[  0   1   1   0   1   0]
            [  0   0   0   0   0   0]
            [  1   0   0   1   0   1]
            [  0   0   0   0   0   0]
            [  0 -12  -8   0  -3   0]
            [  0   0   0   0   0   0]]

    """
    screws = []
    for w, L in zip(screw, radius):
        screw = GetScrew(w, L)
        screws.append(screw)
    return np.array(screws).T


'''________________________________________________________________________________'''


def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3ToVec(so3mat)
    if isZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)
    

'''__________________________________________________________________________________'''


def MatrixExp6(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates

    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat

    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if isZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]



'''__________________________________________________________________________________'''


def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]


'''__________________________________________________________________________________'''



def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]

'''__________________________________________________________________________________'''


def TransInv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.

    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


'''__________________________________________________________________________________'''

def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not isZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not isZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)


'''__________________________________________________________________________________'''

def MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix

    :param R: A matrix in SE3
    :return: The matrix logarithm of R

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[
            np.c_[np.zeros((3, 3)), [T[0][3], T[1][3], T[2][3]]],
            [[0, 0, 0, 0]],
        ]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        V = (
            np.eye(3)
            - omgmat / 2.0
            + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2)
            * np.dot(omgmat, omgmat)
            / theta
        )
        return np.r_[
            np.c_[omgmat, np.dot(V, [T[0][3], T[1][3], T[2][3]])],
            [[0, 0, 0, 0]],
        ]
    

'''__________________________________________________________________________________'''

def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector

    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]

    Used to calculate the Lie bracket [V1, V2] = [adV1]V2

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2,  0,  0,  0],
                  [ 3,  0, -1,  0,  0,  0],
                  [-2,  1,  0,  0,  0,  0],
                  [ 0, -6,  5,  0, -3,  2],
                  [ 6,  0, -4,  3,  0, -1],
                  [-5,  4,  0, -2,  1,  0]])
    """
    omgmat = VecToso3([V[0], V[1], V[2]])
    return np.r_[np.c_[omgmat, np.zeros((3, 3))],
                 np.c_[VecToso3([V[3], V[4], V[5]]), omgmat]]



'''__________________________________________________________________________________'''

def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    """
    Computes the joint angles and velocities at the next timestep using 
    first-order Euler integration.

    :param thetalist: n-vector of current joint variables
    :param dthetalist: n-vector of current joint rates
    :param ddthetalist: n-vector of current joint accelerations
    :param dt: The timestep delta t (e.g., 0.001)
    :return: (thetalistNext, dthetalistNext)
    """

    if dt is None:
        dt = 0.1

    thetalist = np.array(thetalist)
    dthetalist = np.array(dthetalist)
    ddthetalist = np.array(ddthetalist)

    # Next Position: theta_new = theta + velocity * dt
    thetalistNext = thetalist + dthetalist * dt

    # Next Velocity: dtheta_new = dtheta + acceleration * dt
    dthetalistNext = dthetalist + ddthetalist * dt

    return thetalistNext, dthetalistNext

'''__________________________________________________________________________________'''

def parse_spatial_inertias_from_mujoco_xml(xml_path):
    """
    Parses a MuJoCo XML file and returns spatial inertia matrices for each body.

    Returns
    -------
    G_dict : dict
        Dictionary mapping body_name -> 6x6 spatial inertia matrix
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    G_dict = {}

    for body in root.iter('body'):
        body_name = body.get('name')

        inertial = body.find('inertial')
        if inertial is None:
            continue  # skip bodies without inertial data

        # --- Extract inertial parameters ---
        m = float(inertial.get('mass'))

        com = np.fromstring(
            inertial.get('pos', '0 0 0'),
            sep=' ',
            dtype=np.float64
        )

        diag_I = np.fromstring(
            inertial.get('diaginertia'),
            sep=' ',
            dtype=np.float64
        )

        # --- Compute spatial inertia ---
        G = SpatialInertia(m, com, diag_I)

        G_dict[body_name] = G

    return G_dict


def SpatialInertia(m, com, diag_I):
    """
    Calculates the 6x6 spatial inertia matrix Gi in the joint frame.
    Forces all calculations into double precision (float64).
    """
    # Cast inputs to double precision
    m = np.float64(m)
    com = np.array(com, dtype=np.float64)
    diag_I = np.array(diag_I, dtype=np.float64)
    
    x, y, z = com
    I_com = np.diag(diag_I)
    
    # Skew-symmetric matrix [c] for the COM offset
    c_skew = np.array([[0, -z, y], 
                       [z, 0, -x], 
                       [-y, x, 0]], dtype=np.float64)
    
    # Parallel axis theorem: I_joint = I_com - m * [c]^2
    # Note: np.dot on float64 arrays maintains double precision
    I_joint = I_com - m * np.dot(c_skew, c_skew)
    
    G = np.zeros((6, 6), dtype=np.float64)
    G[0:3, 0:3] = I_joint
    G[0:3, 3:6] = m * c_skew
    G[3:6, 0:3] = -m * c_skew
    G[3:6, 3:6] = m * np.eye(3, dtype=np.float64)
    
    return G
