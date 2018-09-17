import cmath
import math
import numpy as np
import sys

this = sys.modules[__name__]


# registers

class Qreg :
    pass

class Creg :
    pass


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
        self.ops = []
        self.qregs = []
        self.cregs = []

    def get_n_qregs(self) :
        return len(self.qregs)

    def get_n_cregs(self) :
        return len(self.cregs)

    def append(self, gate) :
        self.ops.append(gate)

    def allocate_qreg(self, count) :
        qregs = []
        for idx in range(count) :
            qreg = Qreg()
            qregs.append(qreg)
        self.qregs += qregs
        return qregs
    
    def allocate_creg(self, count) :
        cregs = []
        for idx in range(count) :
            creg = Creg()
            cregs.append(creg)
        self.cregs += cregs
        return cregs
    
# Gate

class UnaryGate :

    def __init__(self, qreg) :
        self.in0 = qreg

    def set_matrix(self, mat) :
        self._mat = mat

    def get_matrix(self) :
        return self._mat

class ControlGate :

    def __init__(self, control, target) :
        self.in0 = control
        self.in1 = target

    def set_matrix(self, mat) :
        self._mat = mat

    def get_matrix(self) :
        return self._mat
    

class U(UnaryGate) :
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
    
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

        
class CNot(ControlGate) :
    def __init__(self, control, target) :
        ControlGate.__init__(self, control, target)
        mat = np.array([[0, 1], [1, 0]], np.complex128)
        self.set_matrix(mat)

class Measure :
    def __init__(self, qregs, cregs) :
        self.in0, self.cregs = qregs, cregs

#            
#
#

this.program = Program()


def allocate_qreg(count) :
    return this.program.allocate_qreg(count)
    
def allocate_creg(count) :
    return this.program.allocate_creg(count)

def init_program() :
    this.program = Program()


def current_program() :
    return this.program

def measure(qregs, cregs) :
    qregs, cregs = _arrange_type_2(qregs, cregs)
    measure = Measure(qregs, cregs)
    this.program.append(measure)


# common gates

class H(UnaryGate) :
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        mat = math.sqrt(0.5) * np.array([[1, 1], [1, -1]], np.complex128)
        self.set_matrix(mat)

class X(UnaryGate) :
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        mat = np.array([[0, 1], [1, 0]], np.complex128)
        self.set_matrix(mat)

class Y(UnaryGate) :
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        mat = np.array([[0, -1j], [1j, 0]], np.complex128)
        self.set_matrix(mat)

class Z(UnaryGate) :
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        mat = np.array([[1, 0], [0, -1]], np.complex128)
        self.set_matrix(mat)
        
def u(theta, phi, lambda_, qregs) :
    qregs = _arrange_type(qregs)
    gate = U(qregs)
    gate.set_angles(theta, phi, lambda_)
    this.program.append(gate)

def h(qregs) :
    qregs = _arrange_type(qregs)
    gate = H(qregs)
    this.program.append(gate)

def x(qregs) :
    qregs = _arrange_type(qregs)
    gate = X(qregs)
    this.program.append(gate)

def y(qregs) :
    qregs = _arrange_type(qregs)
    gate = Y(qregs)
    this.program.append(gate)

def z(qregs) :
    qregs = _arrange_type(qregs)
    gate = Z(qregs)
    this.program.append(gate)

def cx(controls, targets) :
    controls, targets = _arrange_type_2(controls, targets)
    gate = CNot(controls, targets)
    this.program.append(gate)
