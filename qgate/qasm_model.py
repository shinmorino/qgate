import cmath
import math
import numpy as np

# registers

class Qreg :
    pass

class Creg :
    pass

#
# global register repositories
#

global_qregs = []
global_cregs = []

def allocate_qreg(count) :
    qregs = []
    for idx in range(count) :
        qreg = Qreg()
        qregs.append(qreg)
        global_qregs.append(qreg)
    return qregs
    
def allocate_creg(count) :
    cregs = []
    for idx in range(count) :
        creg = Creg()
        cregs.append(creg)
        global_cregs.append(creg)
    return cregs


def _arrange_type(obj) :
    if type(obj) is list or type(obj) is tuple :
        return obj
    return [obj]

def _arrange_type_2(obj0, obj1) :
    obj0 = _arrange_type(obj0)
    obj1 = _arrange_type(obj1)
    if len(obj0) != len(obj1) :
        if len(obj0) != 1 and len(obj1) != 1 :
            raise RuntimeError()
    return obj0, obj1


class Program :
    def __init__(self) :
        #self.qregs = []
        #self.cregs = []
        self.ops = []

    def append(self, gate) :
        self.ops.append(gate)
    
# Gate

class UnaryOp :

    def get_n_inputs(self) :
        return 1

    def set_matrix(self, mat) :
        self._mat = mat

class BinaryOp :

    def get_n_inputs(self) :
        return 2


class U(UnaryOp) :
    def __init__(self, qregs) :
        self.in0 = qregs
    
    def set_angles(self, theta = 0, phi = 0, _lambda = 0) :
        self._theta = theta
        self._phi = phi
        self._lambda = _lambda

        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)
        exp_j_phi_plus_lambda_2 = cmath.exp(0.5j * (phi + _lambda))
        exp_j_phi_minus_lambda_2 = cmath.exp(0.5j * (phi - _lambda))
        
        a00 = 1. / exp_j_phi_plus_lambda_2 * cos2
        a01 = - 1. / exp_j_phi_minus_lambda_2 * sin2
        a10 = exp_j_phi_minus_lambda_2 * sin2
        a11 = exp_j_phi_plus_lambda_2 * cos2

        self.set_matrix(mat)

        
class CNot(BinaryOp) :
    def __init__(self, control, target) :
        self.in0, self.in1 = control, target
        
    def get_n_inputs(self) :
        return 2


class Measure(UnaryOp) :
    def __init__(self, qregs, cregs) :
        self.qregs, self.cregs = qregs, cregs
        
    def get_n_inputs(self) :
        return 1

#            
#
#

_program = Program()

def current_program() :
    return _program

def measure(qregs, cregs) :
    qregs, cregs = _arrange_type_2(qregs, cregs)
    measure = Measure(qregs, cregs)
    _program.append(measure)


# common gates

class H(UnaryOp) :
    def __init__(self, qregs) :
        U.__init__(self, qregs)
        mat = math.sqrt(0.5) * np.array([[1, 1], [1, -1]], np.complex128)
        self.set_matrix(mat)

class X(UnaryOp) :
    def __init__(self, qregs) :
        U.__init__(self, qregs)
        mat = np.array([[0, 1], [1, 0]], np.complex128)
        self.set_matrix(mat)

class Y(UnaryOp) :
    def __init__(self, qregs) :
        U.__init__(self, qregs)
        mat = np.array([[0, -1j], [1j, 0]], np.complex128)
        self.set_matrix(mat)

class Z(UnaryOp) :
    def __init__(self, qregs) :
        U.__init__(self, qregs)
        mat = np.array([[1, 0], [0, -1]], np.complex128)
        self.set_matrix(mat)
        
def u(theta, phi, lambda_, qregs) :
    qregs = _arrange_type(qregs)
    gate = U(qregs)
    gate.set_angles(theta, phi, lambda_)
    _program.append(gate)

def h(qregs) :
    qregs = _arrange_type(qregs)
    gate = H(qregs)
    _program.append(gate)

def x(qregs) :
    qregs = _arrange_type(qregs)
    gate = X(qregs)
    _program.append(gate)

def y(qregs) :
    qregs = _arrange_type(qregs)
    gate = Y(qregs)
    _program.append(gate)

def z(qregs) :
    qregs = _arrange_type(qregs)
    gate = Z(qregs)
    _program.append(gate)

def cx(controls, targets) :
    controls, targets = _arrange_type_2(controls, targets)
    gate = CNot(controls, targets)
    _program.append(gate)
