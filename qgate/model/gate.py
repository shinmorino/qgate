import numpy as np
import math
import cmath


def adjoint(mat) :
    return np.conjugate(mat)

#
# builtin gate
#

#
# built-in gate implementation
#

from .model import UnaryGate, ControlGate

class U(UnaryGate) :
    @staticmethod
    def mat(theta, phi, _lambda) :
        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)

        # Ref: https://quantumexperience.ng.bluemix.net/qx/tutorial?sectionId=full-user-guide&page=002-The_Weird_and_Wonderful_World_of_the_Qubit~2F004-advanced_qubit_gates
        a00 =                                      cos_theta_2
        a01 = - cmath.exp(1.j * _lambda)         * sin_theta_2
        a10 =   cmath.exp(1.j * phi)             * sin_theta_2
        a11 =   cmath.exp(1.j * (_lambda + phi)) * cos_theta_2
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, theta, phi, _lambda, qregs) :
        UnaryGate.__init__(self, qregs)
        self._theta = theta
        self._phi = phi
        self._lambda = _lambda
        self.set_matrix(U.mat(theta, phi, _lambda))
            
class U2(UnaryGate) :
    # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    @staticmethod
    def mat(phi, _lambda) :
        a00 =   1.
        a01 = - cmath.exp(1.j * _lambda)
        a10 =   cmath.exp(1.j * phi)
        a11 =   cmath.exp(1.j * (_lambda + phi))
        return math.sqrt(0.5) * np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, phi, lambda_, qregs) :
        UnaryGate.__init__(self, qregs)
        self.phi = phi
        self._lambda = lambda_
        self.set_matrix(U2.mat(phi, lambda_))
            
class U1(UnaryGate) :
    # gate u1(lambda) q { U(0,0,lambda) q; }
    @staticmethod
    def mat(_lambda) :
        a00 =   1.
        a01 =   0.
        a10 =   0.
        a11 =   cmath.exp(1.j * _lambda)
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, lambda_, qregs) :
        UnaryGate.__init__(self, qregs)
        self._lambda = lambda_
        self.set_matrix(U1.mat(lambda_))
            
class CX(ControlGate) :
    mat = np.array([[0, 1], [1, 0]], np.complex128)
    def __init__(self, control, target) :
        ControlGate.__init__(self, control, target)
        self.set_matrix(CX.mat)
            
class ID(UnaryGate) :
    mat = np.array([[1, 0], [0, 1]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)

class X(UnaryGate) :
    mat = np.array([[0, 1], [1, 0]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(X.mat)

class Y(UnaryGate) :
    mat = np.array([[0, -1j], [1j, 0]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(Y.mat)

class Z(UnaryGate) :
    mat = np.array([[1, 0], [0, -1]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(Z.mat)

class H(UnaryGate) :
    mat = math.sqrt(0.5) * np.array([[1, 1], [1, -1]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(H.mat)

class S(UnaryGate) :
    mat = np.array([[1, 0], [0, 1j]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(SDG.mat)

class T(UnaryGate) :
    mat = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4.)]], np.complex128)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(T.mat)

class RX(UnaryGate) :
    @staticmethod
    def mat(theta) :
        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)
        a00, a01 =         cos_theta_2, - 1j  * sin_theta_2
        a10, a11 = - 1j  * sin_theta_2,         cos_theta_2
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, theta, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(RX.mat)
        
class RY(UnaryGate) :
    @staticmethod
    def mat(theta) :
        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)
        a00, a01 = cos_theta_2, - sin_theta_2
        a10, a11 = sin_theta_2,   cos_theta_2
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, theta, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(RY.mat(theta))

class RZ(UnaryGate) :
    mat = U1.mat # U1.mat(phi)
    def __init__(self, phi, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(RX.mat(phi))
