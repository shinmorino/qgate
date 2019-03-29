from qgate.model import GateType
import qgate.model.gate_type as gtype
import numpy as np
import math
import cmath

import sys
if sys.version_info[0] < 3:
    def _attach(clsobj, method) :
        from types import MethodType
        clsobj.pymat = MethodType(method, None, clsobj)
    
else :
    def _attach(clsobj, method) :
        setattr(clsobj, 'pymat', method)

# U
def U_mat(self) :
    theta, phi, _lambda = self.args
    
    theta2 = theta / 2.
    cos_theta_2 = math.cos(theta2)
    sin_theta_2 = math.sin(theta2)

    # Ref: https://quantumexperience.ng.bluemix.net/qx/tutorial?sectionId=full-user-guide&page=002-The_Weird_and_Wonderful_World_of_the_Qubit~2F004-advanced_qubit_gates
    a00 =                                      cos_theta_2
    a01 = - cmath.exp(1.j * _lambda)         * sin_theta_2
    a10 =   cmath.exp(1.j * phi)             * sin_theta_2
    a11 =   cmath.exp(1.j * (_lambda + phi)) * cos_theta_2
    return np.array([[a00, a01], [a10, a11]], np.complex128)
_attach(gtype.U, U_mat)
            
# gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
def U2_mat(self) :
    phi, _lambda = self.args
    a00 =   1.
    a01 = - cmath.exp(1.j * _lambda)
    a10 =   cmath.exp(1.j * phi)
    a11 =   cmath.exp(1.j * (_lambda + phi))
    return math.sqrt(0.5) * np.array([[a00, a01], [a10, a11]], np.complex128)
_attach(gtype.U2, U2_mat)

# gate u1(lambda) q { U(0,0,lambda) q; }
def U1_mat(self) :
    _lambda,  = self.args
    a00 =   1.
    a01 =   0.
    a10 =   0.
    a11 =   cmath.exp(1.j * _lambda)
    return np.array([[a00, a01], [a10, a11]], np.complex128)
_attach(gtype.U1, U1_mat)

# ID        
def ID_mat(self) :
    return ID_mat.mat
ID_mat.mat = np.array([[1, 0], [0, 1]], np.complex128)
_attach(gtype.ID, ID_mat)

# X
def X_mat(self) :
    return X_mat.mat
X_mat.mat = np.array([[0, 1], [1, 0]], np.complex128)
_attach(gtype.X, X_mat)


# Y
def Y_mat(self) :
    return Y_mat.mat
Y_mat.mat = np.array([[0, -1j], [1j, 0]], np.complex128)
_attach(gtype.Y, Y_mat)

# Z
def Z_mat(self) :
    return Z_mat.mat
Z_mat.mat = np.array([[1, 0], [0, -1]], np.complex128)
_attach(gtype.Z, Z_mat)

# H
def H_mat(self) :
    return H_mat.mat
H_mat.mat = math.sqrt(0.5) * np.array([[1, 1], [1, -1]], np.complex128)
_attach(gtype.H, H_mat)

# S
def S_mat(self) :
    return S_mat.mat
S_mat.mat = np.array([[1, 0], [0, 1j]], np.complex128)
_attach(gtype.S, S_mat)

# T
def T_mat(self) :
    return T_mat.mat
T_mat.mat = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4.)]], np.complex128)
_attach(gtype.T, T_mat)

# RX
def RX_mat(self) :
    theta,  = self.args
    theta2 = theta / 2.
    cos_theta_2 = math.cos(theta2)
    sin_theta_2 = math.sin(theta2)
    a00, a01 =         cos_theta_2, - 1j  * sin_theta_2
    a10, a11 = - 1j  * sin_theta_2,         cos_theta_2
    return np.array([[a00, a01], [a10, a11]], np.complex128)
_attach(gtype.RX, RX_mat)

# RY
def RY_mat(self) :
    theta,  = self.args
    theta2 = theta / 2.
    cos_theta_2 = math.cos(theta2)
    sin_theta_2 = math.sin(theta2)
    a00, a01 = cos_theta_2, - sin_theta_2
    a10, a11 = sin_theta_2,   cos_theta_2
    return np.array([[a00, a01], [a10, a11]], np.complex128)
_attach(gtype.RY, RY_mat)

# RZ
def RZ_mat(self) :
    theta,  = self.args
    theta2 = theta / 2.
    d0 = cmath.exp(-1.j * theta2)
    d1 = cmath.exp( 1.j * theta2)
    return np.array([[d0, 0.], [0., d1]], np.complex128)
_attach(gtype.RZ, RZ_mat)

# Exp
def ExpiI_mat(self) :
    theta,  = self.args
    d = cmath.exp(1j * theta)
    return np.array([[d, 0], [0, d]], np.complex128)
_attach(gtype.ExpiI, ExpiI_mat)

# Exp
def ExpiZ_mat(self) :
    theta,  = self.args
    d0 = cmath.exp(1j * theta)
    d1 = cmath.exp(-1j * theta)
    return np.array([[d0, 0], [0, d1]], np.complex128)
_attach(gtype.ExpiZ, ExpiZ_mat)

# utility
def SH_mat(self) :
    return SH_mat.mat
SH_mat.mat = math.sqrt(0.5) * np.array([[1., 1.], [1.j, -1.j]], np.complex128)
_attach(gtype.SH, SH_mat)
