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
    def __init__(self, creg_array, id, idx) :
        self.creg_array = creg_array
        self.id = id
        self.idx = idx

    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, Creg) :
            return self.id == other.id
        return False


class CregArray :
    def __init__(self, id, creg_id_offset, length) :
        self.id = id
        self.cregs = []
        for idx in range(length) :
            self.cregs.append(Creg(self, creg_id_offset + idx, idx))

    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, CregArray) :
            return self.id == other.id
        return False
    
    def get_cregs(self) :
        return self.cregs

    def length(self) :
        return len(self.cregs)
    

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
    def __init__(self) :
        self.idx = - 1

    def set_idx(self, idx) :
        self.idx = idx

    def get_idx(self) :
        return self.idx
    

class UnaryGate(Operator) :

    def __init__(self, qreg) :
        Operator.__init__(self)
        self.in0 = _arrange_type(qreg)

    def set_matrix(self, mat) :
        self._mat = mat

    def get_matrix(self) :
        return self._mat

class ControlGate(Operator) :

    def __init__(self, control, target) :
        Operator.__init__(self)
        self.in0, self.in1 = _arrange_type_2(control, target)

    def set_matrix(self, mat) :
        self._mat = mat

    def get_matrix(self) :
        return self._mat
    
class Measure(Operator) :
    def __init__(self, qregs, cregs) :
        Operator.__init__(self)
        self.in0, self.cregs = _arrange_type_2(qregs, cregs)


class Barrier(Operator) :
    def __init__(self, qregslist) :
        self.qregset = set()
        for qregs in qregslist :
            if type(qregs) is list or type(qregs) is tuple :
               self.qregset |= set(qregs)
            else :
                self.qregset.add(qregs)

class Reset(Operator) :
    def __init__(self, qregslist) :
        Operator.__init__(self)
        self.qregset = set()
        for qregs in qregslist :
            if type(qregs) is list or type(qregs) is tuple :
               self.qregset |= set(qregs)
            else :
                self.qregset.add(qregs)
                
               
class Clause(Operator) :
    def __init__(self) :
        Operator.__init__(self)
        self.ops = []
        self.qregs = set()

    def set_qregs(self, qregs) :
        self.qregs = qregs

    def get_qregs(self) :
        return self.qregs

    def add_op(self, op) :
        assert isinstance(op, Operator), "op is not an operator."
        self.ops.append(op)

class IfClause(Operator) :
    def __init__(self, creg_array, val) :
        Operator.__init__(self)
        self.creg_array = creg_array
        self.val = val

    def set_clause(self, clause) :
        self.clause = clause
        
        
class IsolatedClauses(Operator) :
    def __init__(self) :
        Operator.__init__(self)
        self.clauses = []
                
    def append(self, circuit) :
        self.clauses.append(circuit)
        

class Program :
    def __init__(self) :
        self.circuit = Clause()
        self.qregs = set()
        self.creg_arrays = set()
        self.cregs = set()
        
    def get_n_qregs(self) :
        return len(self.qregs)

    def get_n_cregs(self) :
        return len(self.cregs)

    def set_regs(self, qregs, creg_arrays, cregs) :
        self.qregs = qregs
        self.creg_arrays = creg_arrays
        self.cregs = cregs
    
    def add_op(self, op) :
        self.circuit.add_op(op)

    def get_circuits(self) :
        if isinstance(self.circuit, IsolatedClauses) :
            return self.circuit.clauses
        else :
            return [self.circuit]

    def allocate_qreg(self, count) :
        qregs = []
        for idx in range(count) :
            qreg = Qreg(len(self.qregs))
            qregs.append(qreg)
            self.qregs.add(qreg)
        return qregs
    
    def allocate_creg(self, count) :
        creg_array = CregArray(len(self.creg_arrays), len(self.cregs), count)
        self.creg_arrays.add(creg_array)
        self.cregs |= set(creg_array.get_cregs())
        return creg_array.get_cregs()

#
# builtin gate
#

#
# built-in gate implementation
#
    
class U(UnaryGate) :
    def __init__(self, theta, phi, _lambda, qregs) :
        UnaryGate.__init__(self, qregs)
        self._set(theta, phi, _lambda)
    
    def _set(self, theta, phi, _lambda) :
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
        mat = np.matrix([[0, 1], [1, 0]], np.complex128)
        self.set_matrix(mat)
