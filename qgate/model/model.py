import numpy as np

# quantum register
class Qreg :
    count = 0
    def __init__(self) :
        self.id = Qreg.count
        Qreg.count += 1

    # FIXME: implement
    def __del__(self) :
        pass
        
    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, Qreg) :
            return self.id == other.id
        return False

    
# Value reference, for both in and out.
class Reference :
    count = 0
    
    def __init__(self) :
        self.id = Reference.count
        Reference.count += 1;

    def __hash__(self) :
        return self.id

    def __eq__(self, other) :
        if isinstance(other, Reference) :
            return self.id == other.id
        return False    


class Operator :
    def __init__(self) :
        self.idx = - 1

    def set_idx(self, idx) :
        self.idx = idx

    def get_idx(self) :
        return self.idx


class GateType :
    def __init__(self, *args) :
        self.args = args
        
class Gate(Operator) :

    def __init__(self, gate_type) :
        Operator.__init__(self)
        self.gate_type = gate_type
        self.adjoint = False
        self.qreglist = None
        self.cntrlist = None

    def set_adjoint(adjoint) :
        self.adjoint = adjoint

    def set_control(self, cntrlist) :
        assert self.cntrlist is None, 'cntr args already set.'
        self.cntrlist = [cntrlist] if isinstance(cntrlist, Qreg) else cntrlist

    def set_qreglist(self, qreglist) :
        assert self.qreglist is None, 'qreg list already set.'
        self.qreglist = qreglist

    def check_constraints(self) :
        self.gate_type.constraints(self)
        
    # FIXME: rename
    def create(self, qreglist, cntrlist) :
        obj = Gate(self.gate_type)
        obj.set_control(cntrlist)
        obj.set_qreglist(qreglist)
        return obj

class ComposedGate(Operator) :

    def __init__(self) :
        Operator.__init__(self)
        self.type_list = []
        self.qreglist = None
        self.cntrlist = None

    def add(self, type_list) :
        self.type_list = type_list
        
    def set_control(self, cntrlist) :
        assert self.cntrlist is None, 'cntr args already set.'
        self.cntrlist = [cntrlist] if isinstance(cntrlist, Qreg) else cntrlist

    def set_qreglist(self, qreglist) :
        assert self.qreglist is None, 'qreg list already set.'
        self.qreglist = qreglist

    def get_clause(self) :
        return None
        
    # FIXME: rename
    def create(self, qreglist, cntrlist) :
        obj = Gate(self.gate_type)
        obj.set_control(cntrlist)
        obj.set_qreglist(qreglist)
        return obj
    
    
class Measure(Operator) :
    def __init__(self, qreg, ref) :
        if not isinstance(qreg, Qreg) or not isinstance(ref, Reference) :
            raise RuntimeError('Wrong argument for Measure, {}, {}.'.format(repr(qreg), repr(ref)))
        Operator.__init__(self)
        self.qreg, self.outref = qreg, ref


class Barrier(Operator) :
    def __init__(self, qregset) :
        Operator.__init__(self)
        assert all([isinstance(qreg, Qreg) for qreg in qregset]), 'arguments must be Qreg.'
        self.qregset = set(qregset)

class Reset(Operator) :
    def __init__(self, qregset) :
        Operator.__init__(self)
        assert all([isinstance(qreg, Qreg) for qreg in qregset]), 'arguments must be Qreg.'
        self.qregset = set(qregset)
                
               
class Clause(Operator) :
    def __init__(self) :
        Operator.__init__(self)
        self.ops = []
        self.qregset = set()
        self.refset = set()

    def set_qregset(self, qregset) :
        self.qregset = qregset

    def get_qregset(self) :
        return self.qregset

    def set_refset(self, refs) :
        self.refset = set(refs)

    def get_refset(self) :
        return self.refset

    def add_op(self, op) :
        assert isinstance(op, Operator), "Unknown argument, {}.".format(repr(op))
        self.ops.append(op)

    def add(self, *args) :
        for arg in args :
            if isinstance(arg, (list, tuple, set)) :
                clause = Clause()
                clause.add(*arg)
                self.ops.append(clause)
            elif isinstance(arg, (Qreg, Operator)) :
                self.ops.append(arg)
            else :
                assert False, 'Unknown argument, {}'.format(repr(arg))
        
# can be top-level clause.
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

    def set_refset(self, refs) :
        self.refset = set(refs)

    def get_refset(self) :
        return self.refset

    def __len__(self) :
        return len(self.clauses)
    
    def __iter__(self) :
        return iter(self.clauses)

class IfClause(Operator) :
    def __init__(self, refs, val) :
        Operator.__init__(self)
        self.refs = refs
        self.val = val

    def set_clause(self, clause) :
        self.clause = clause
