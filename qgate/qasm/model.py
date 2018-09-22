import cmath
import math
import numpy as np


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
class Operator :
    regfuncs = []

    @staticmethod
    def add_regfunc(regfunc) :
        Operator.regfuncs.append(regfunc)
    @staticmethod
    def del_regfunc(regfunc) :
        Operator.regfuncs.remove(regfunc)
    
    @staticmethod
    def register(op) :
        for func in Operator.regfuncs :
            func(op)


class UnaryGate(Operator) :

    def __init__(self, qreg) :
        self.in0 = _arrange_type(qreg)
        Operator.register(self)

    def set_matrix(self, mat) :
        self._mat = mat

    def get_matrix(self) :
        return self._mat

class ControlGate(Operator) :

    def __init__(self, control, target) :
        self.in0, self.in1 = _arrange_type_2(control, target)
        Operator.register(self)

    def set_matrix(self, mat) :
        self._mat = mat

    def get_matrix(self) :
        return self._mat
    
class Measure(Operator) :
    def __init__(self, qregs, cregs) :
        self.in0, self.cregs = _arrange_type_2(qregs, cregs)
        Operator.register(self)


class Barrier(Operator) :
    def __init__(self, qregslist) :
        self.ops = set()
        for qregs in qregslist :
            if type(qregs) is list or type(qregs) is tuple :
               self.ops |= set(qregs)
            else :
                self.ops.add(qregs)
        Operator.register(self)


#class If(Operator) :
#    def __init__(self, val, ops) :
#        pass
               
class Circuit :
    def __init__(self) :
        self.ops = []
        self.qregs = set()
        self.cregs = set()

    def set_regs(self, qregs, cregs) :
        self.qregs = qregs
        self.cregs = cregs

    def get_regs(self) :
        return self.qregs, self.cregs

    def add_op(self, op) :
        self.ops.append(op)
        if isinstance(op, Measure) :
            self.qregs |= set(op.in0)
            self.cregs |= set(op.cregs)
        elif isinstance(op, UnaryGate) :
            self.qregs |= set(op.in0)
        elif isinstance(op, ControlGate) :
            self.qregs |= set(op.in0 + op.in1)
        # FIXME: add barrier

        
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
# builtin gate
#

#
# built-in gate implementation
#
    
class U(UnaryGate) :
    def __init__(self, theta, phi, _lambda, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set(theta, phi, _lambda)
    
    def set(self, theta = 0, phi = 0, _lambda = 0) :
        self._theta = theta
        self._phi = phi
        self._lambda = _lambda

        theta2 = theta / 2.
        cos_theta_2 = math.cos(theta2)
        sin_theta_2 = math.sin(theta2)
        exp_j_phi_plus_lambda_2 = cmath.exp(0.5j * (phi + _lambda))
        exp_j_phi_minus_lambda_2 = cmath.exp(0.5j * (phi - _lambda))
        
        a00 = 1. / exp_j_phi_plus_lambda_2 * cos_theta_2
        a01 = - 1. / exp_j_phi_minus_lambda_2 * sin_theta_2
        a10 = exp_j_phi_minus_lambda_2 * sin_theta_2
        a11 = exp_j_phi_plus_lambda_2 * cos_theta_2

        mat = np.matrix([[a00, a01], [a10, a11]], np.complex128)
        self.set_matrix(mat)

        
class CX(ControlGate) :
    def __init__(self, control, target) :
        ControlGate.__init__(self, control, target)
        mat = np.array([[0, 1], [1, 0]], np.complex128)
        self.set_matrix(mat)


class Clause :
    def __init__(self, *gates) :
        self.gates = gates
        Operator.unregister(self.gates)
        Operator.register(self)
        
def clause(*gates) :
    return Clause(gates)
