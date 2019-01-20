import cmath
import math
import numpy as np

def adjoint(mat) :
    return np.conjugate(mat)


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

    def __getitem__(self, key) :
        return self.cregs[key]

    def __len__(self) :
        return len(self.cregs)

    def __iter__(self) :
        return iter(self.cregs)
    

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
        if isinstance(cregs, CregArray) :
            cregs = list(cregs)
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
        self.qregset = set()

    def set_qregset(self, qregset) :
        self.qregset = qregset

    def get_qregset(self) :
        return self.qregset

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
                
    def append(self, clause) :
        self.clauses.append(clause)
        

class Program :
    def __init__(self) :
        self.clause = Clause()
        self.qregset = set()
        self.creg_arrays = set()
        self.cregset = set()
        
    def get_n_qregs(self) :
        return len(self.qregset)

    def get_n_cregs(self) :
        return len(self.cregset)

    def set_regs(self, qregs, creg_arrays, cregs) :
        self.qregset = set(qregs)
        self.creg_arrays = creg_arrays
        self.cregset = set(cregs)
    
    def add_op(self, op) :
        self.clause.add_op(op)

    def get_circuits(self) :
        if isinstance(self.clause, IsolatedClauses) :
            return self.clause.clauses
        else :
            return [self.clause]

    def allocate_qreg(self, count) :
        qregs = []
        for idx in range(count) :
            qreg = Qreg(len(self.qregset))
            qregs.append(qreg)
            self.qregset.add(qreg)
        return qregs
    
    def allocate_creg(self, count) :
        creg_array = CregArray(len(self.creg_arrays), len(self.cregset), count)
        self.creg_arrays.add(creg_array)
        self.cregset |= set(creg_array)
        return creg_array

    
#
# builtin gate
#

#
# built-in gate implementation
#
    
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
            
# functions to instantiate operators

def measure(qregs, cregs) :
    return Measure(qregs, cregs)

def barrier(*qregs) :
    bar = Barrier(qregs)
    return bar

def reset(*qregs) :
    reset = Reset(qregs)
    return reset
        
def clause(*ops) :
    cl = Clause()
    for op in ops :
        cl.add_op(op)
    return cl

def if_(creg_array, val, ops) :
    if_clause = IfClause(creg_array[0].creg_array, val)
    cl = clause(ops)
    if_clause.set_clause(cl)
    return if_clause
