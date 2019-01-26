import numpy as np

# registers

class Qreg :
    count = 0
    
    def __init__(self) :
        self.id = Qreg.count
        Qreg.count += 1
        
    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, Qreg) :
            return self.id == other.id
        return False

class Creg :
    count = 0
    
    def __init__(self) :
        self.id = Creg.count
        Creg.count += 1;

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
        self.qregset = set()
        self.cregset = set()

    def set_qregset(self, qregset) :
        self.qregset = qregset

    def get_qregset(self) :
        return self.qregset

    def set_cregset(self, cregset) :
        self.cregset = cregset

    def get_cregset(self) :
        return self.cregset

    def add_op(self, op) :
        assert isinstance(op, Operator), "op is not an operator."
        self.ops.append(op)

    def add(self, *args) :
        for obj in args :
            if isinstance(obj, Operator) :
                self.add_op(obj)
            elif isinstance(obj, Qreg) :
                self.qregset |= { obj }
            elif isinstance(obj, list) :
                self.add(*obj)
            else :
                raise RuntimeError('Unknown object added.')
        

class ClauseList(Operator) :
    def __init__(self) :
        Operator.__init__(self)
        self.clauses = []
                
    def append(self, clause) :
        self.clauses.append(clause)

    def set_qregset(self, qregset) :
        self.qregset = qregset

    def get_qregset(self) :
        return self.qregset

    def set_cregset(self, cregset) :
        self.cregset = cregset

    def get_cregset(self) :
        return self.cregset

    def __len__(self) :
        return len(self.clauses)
    
    def __iter__(self) :
        return iter(self.clauses)

class IfClause(Operator) :
    def __init__(self, creg_array, val) :
        Operator.__init__(self)
        self.creg_array = creg_array
        self.val = val

    def set_clause(self, clause) :
        self.clause = clause
