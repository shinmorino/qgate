import cmath
import math
import numpy as np
import sys

this = sys.modules[__name__]


# registers

class Qreg :
    def __init__(self, id) :
        self.id = id
        
    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, Qreg) :
            return self.id == other.id
        return False
        

class Creg :
    def __init__(self, id) :
        self.id = id

    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, Creg) :
            return self.id == other.id
        return False

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
    

class Circuit :
    def __init__(self) :
        self.ops = []
        self.qregs = set()
        self.cregs = set()
        
    def get_n_qregs(self) :
        return len(self.qregs)

    def get_n_cregs(self) :
        return len(self.cregs)

    def set_regs(self, qregs, cregs) :
        self.qregs = qregs
        self.cregs = cregs

    def add_op(self, op) :
        self.ops.append(op)
        if isinstance(op, Measure) :
            self.qregs |= set(op.in0)
            self.cregs |= set(op.cregs)
        elif isinstance(op, UnaryGate) :
            self.qregs |= set(op.in0)
        elif isinstance(op, ControlGate) :
            self.qregs |= set(op.in0 + op.in1)

    def get_qreg_lane(self, qreg) :
        return list(self.qregs).index(qreg)
    
    def get_creg_lane(self, creg) :
        return list(self.cregs).index(creg)
        
    
        
class Program :
    def __init__(self) :
        self.circuits = []
        self.qregs = set()
        self.cregs = set()
        
    def get_n_qregs(self) :
        return len(self.qregs)

    def get_n_cregs(self) :
        return len(self.cregs)

    def set_regs(self, qregs, cregs) :
        self.qregs = qregs.copy()
        self.cregs = cregs.copy()
    
    def add_circuit(self, circuit) :
        self.circuits.append(circuit)

    def add_op(self, op) :
        self.circuits[0].add_op(op)

    def allocate_qreg(self, count) :
        qregs = []
        for idx in range(count) :
            qreg = Qreg(len(self.qregs))
            qregs.append(qreg)
            self.qregs.add(qreg)
        return qregs
    
    def allocate_creg(self, count) :
        cregs = []
        for idx in range(count) :
            creg = Creg(len(self.cregs))
            cregs.append(creg)
            self.cregs.add(creg)
        return cregs



#
# Gate implementations
#
    
class NullGate(UnaryGate) :
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        
    
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
# module-level interface
#


def allocate_qreg(count) :
    return this.program.allocate_qreg(count)
    
def allocate_creg(count) :
    return this.program.allocate_creg(count)

def init_program() :
    this.program = Program()
    this.program.add_circuit(Circuit())
    
def current_program() :
    return this.program

def measure(qregs, cregs) :
    qregs, cregs = _arrange_type_2(qregs, cregs)
    measure = Measure(qregs, cregs)
    this.program.add_op(measure)


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
    this.program.add_op(gate)

def h(qregs) :
    qregs = _arrange_type(qregs)
    gate = H(qregs)
    this.program.add_op(gate)

def x(qregs) :
    qregs = _arrange_type(qregs)
    gate = X(qregs)
    this.program.add_op(gate)

def y(qregs) :
    qregs = _arrange_type(qregs)
    gate = Y(qregs)
    this.program.add_op(gate)

def z(qregs) :
    qregs = _arrange_type(qregs)
    gate = Z(qregs)
    this.program.add_op(gate)

def cx(controls, targets) :
    controls, targets = _arrange_type_2(controls, targets)
    gate = CNot(controls, targets)
    this.program.add_op(gate)



# module level
init_program()
