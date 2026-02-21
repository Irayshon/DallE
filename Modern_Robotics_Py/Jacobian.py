from Modern_Robotics_Py.Tools import *
import numpy as np

'''__________________________________________________________________________________'''


def Jacobian(list, thetalist, frame='space'):
    """
    Computes the Jacobian (Space or Body) for an open chain robot.
    
    :param list: The joint screw axes (Slist or Blist) at home position (6xn matrix)
    :param thetalist: A list of joint coordinates
    :param frame: 'space' for Space Jacobian, 'body' for Body Jacobian
    :return: The 6xn Jacobian matrix

    Example Input:
        Slist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
        frame = 'space'
    Output:
        np.array([[  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]
                  [  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])


    """
    J = np.array(list).copy().astype(float)
    T = np.eye(4)
    n = len(thetalist)

    if frame.lower() == 'space':
        ''' Space Jacobian: Js_i = Ad_{e^[S1]q1 ... e^[S_i-1]q_i-1} (Si)
            Moving from base to tip'''
        for i in range(1, n):
            T = np.dot(T, MatrixExp6( VecTose3(np.array(list)[:, i - 1] * thetalist[i - 1])))
            J[:, i] = np.dot( Adjoint(T), np.array(list)[:, i])
            
    elif frame.lower() == 'body':
        '''Body Jacobian: Jb_i = Ad_{e^-[Bn]qn ... e^-[B_i+1]q_i+1} (Bi)
           Moving from tip to base'''
        for i in range(n - 2, -1, -1):
            T = np.dot(T,  MatrixExp6( VecTose3(np.array(list)[:, i + 1] * -thetalist[i + 1])))
            J[:, i] = np.dot( Adjoint(T), np.array(list)[:, i])
            
    else:
        raise ValueError("Frame must be 'space' or 'body'")

    return J

'''__________________________________________________________________________________'''


def ConvertJacobianFrame(J, T, target_frame='space'):
    """
    Converts a Jacobian from one frame to another.
    
    :param J: The input Jacobian matrix (6xn)
    :param T: The transformation matrix (4x4) 
    :param target_frame: 'space' to convert Jb to Js, 
                         'body' to convert Js to Jb
    :return: The converted 6xn Jacobian matrix
  
    Example Input:
        J = np.array([[  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]
                  [  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])
        T = np.array([[ 0.  1.  0. -4.]
                        [-0.  0. -1. -3.]
                        [-1.  0.  0. 10.]
                        [ 0.  0.  0.  1.]])
        target_frame = 'space'
    Output:
        np.array[[-0.  0.  0. -0.  1.  0.]
                [ 0.  1.  1.  0.  0. -1.]
                [ 1. -0. -0.  1.  0.  0.]
                [-0. -5. -5. -0. -0. 10.]
                [-0.  0. -0.  4. 10.  0.]
                [ 0.  0. -4. -0. -0.  4.]]
    """

    if target_frame.lower() == 'space':
        '''Formula: Js = [Ad_Tsb] * Jb
           Input T should be Tsb (Transformation from space to body)'''
        return np.dot( Adjoint(T), J)
        
    elif target_frame.lower() == 'body':
        ''' Formula: Jb = [Ad_Tbs] * Js
            Input T should be Tbs (Transformation from body to space)
            Note: Ad(Tbs) is the same as Inverse(Ad(Tsb))'''
        T = np.linalg.inv(T)
        return np.dot( Adjoint(T), J)
    
    else:
        raise ValueError("target_frame must be 'space' or 'body'")
    

'''__________________________________________________________________________________'''

