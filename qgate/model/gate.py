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

from .model import GateType

class U(GateType) :
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
    
    def __init__(self, theta, phi, _lambda) :
        GateType.__init__(self, theta, phi, _lambda);
            
class U2(GateType) :
    # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    @staticmethod
    def mat(phi, _lambda) :
        a00 =   1.
        a01 = - cmath.exp(1.j * _lambda)
        a10 =   cmath.exp(1.j * phi)
        a11 =   cmath.exp(1.j * (_lambda + phi))
        return math.sqrt(0.5) * np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, phi, _lambda) :
        GateType.__init__(self, phi,_lambda)
            
class U1(GateType) :
    # gate u1(lambda) q { U(0,0,lambda) q; }
    @staticmethod
    def mat(_lambda) :
        a00 =   1.
        a01 =   0.
        a10 =   0.
        a11 =   cmath.exp(1.j * _lambda)
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, _lambda) :
        GateType.__init__(self, _lambda)
        
        
class ID(GateType) :
    def mat() :
        return np.array([[1, 0], [0, 1]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class X(GateType) :
    def mat() :
        return np.array([[0, 1], [1, 0]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class Y(GateType) :
    def mat() :
        return np.array([[0, -1j], [1j, 0]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class Z(GateType) :
    def mat() :
        return np.array([[1, 0], [0, -1]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class H(GateType) :
    def mat() :
        return math.sqrt(0.5) * np.array([[1, 1], [1, -1]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class S(GateType) :
    def mat() :
        return np.array([[1, 0], [0, 1j]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class T(GateType) :
    def mat() :
        return np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4.)]], np.complex128)
    def __init__(self) :
        GateType.__init__(self)

class RX(GateType) :
    @staticmethod
    def mat(theta) :
        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)
        a00, a01 =         cos_theta_2, - 1j  * sin_theta_2
        a10, a11 = - 1j  * sin_theta_2,         cos_theta_2
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, theta) :
        GateType.__init__(self, theta)
        
class RY(GateType) :
    @staticmethod
    def mat(theta) :
        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)
        a00, a01 = cos_theta_2, - sin_theta_2
        a10, a11 = sin_theta_2,   cos_theta_2
        return np.array([[a00, a01], [a10, a11]], np.complex128)
    
    def __init__(self, theta) :
        GateType.__init__(self, theta)

class RZ(GateType) :
    mat = U1.mat # U1.mat(phi)
    def __init__(self, phi) :
        GateType.__init__(self, phi)
