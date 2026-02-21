from Modern_Robotics_Py.Tools import *
import numpy as np

'''________________________________________________________________________________'''

def FKin(M, list, thetalist, frame = 'space'):
    """Computes forward kinematics in the body frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Body Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
        body = True

    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """

    T = np.array(M)
    n = len(thetalist)
    
    if frame.lower() == 'body' :
        """ Body Frame: T = M * Exp(B1) * Exp(B2) ...
            Loop 0 to n-1, multiply on the RIGHT """
        for i in range(n):
            T = np.dot(T,   MatrixExp6(  VecTose3(np.array(list)[:, i] * thetalist[i])))
    elif frame.lower() == 'space':
        """Space Frame: T = Exp(S1) * ... * Exp(Sn) * M
           Loop n-1 down to 0, multiply on the LEFT """
        for i in range(n - 1, -1, -1):
            T = np.dot(  MatrixExp6(  VecTose3(np.array(list)[:, i] * thetalist[i])), T)
            
    return T

'''________________________________________________________________________________'''

def GetFinalPose(T):
    """
    Extracts position and orientation from a 4x4 Transformation matrix as a Quaternion.
    
    :param T: 4x4 Homogeneous transformation matrix
    :return: 1D Numpy array [x, y, z, qx, qy, qz, qw]

    Input:
       [[ 0.  1.  0. -4.]
        [ 0.  0. -1. -3.]
        [-1.  0.  0. 10.]
        [ 0.  0.  0.  1.]]

    Output:
        [-4.  -3.  10.   0.5  0.5 -0.5  0.5]

    """
    position = T[:3, 3]
    R = T[:3, :3]

    '''--- Quaternion Extraction (Shepperd's Method) ---
       This method is numerically stable for any rotation matrix '''
    
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 
             (R[0, 2] - R[2, 0]) / S, 
             (R[1, 0] - R[0, 1]) / S, 
             0.25 * S]
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * S, 
             (R[0, 1] + R[1, 0]) / S, 
             (R[0, 2] + R[2, 0]) / S, 
             (R[2, 1] - R[1, 2]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / S, 
             0.25 * S, 
             (R[1, 2] + R[2, 1]) / S, 
             (R[0, 2] - R[2, 0]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / S, 
             (R[1, 2] + R[2, 1]) / S, 
             0.25 * S, 
             (R[1, 0] - R[0, 1]) / S]

    return np.concatenate((position, q))


'''________________________________________________________________________________'''
