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
        self.ctrllist = None

    def set_adjoint(self, adjoint) :
        self.adjoint = adjoint

    def set_ctrllist(self, ctrllist) :
        self.ctrllist = [ctrllist] if isinstance(ctrllist, Qreg) else ctrllist
        assert all([isinstance(qreg, Qreg) for qreg in self.ctrllist]), 'arguments must be Qreg.'
        
    def set_qreglist(self, qreglist) :
        self.qreglist = qreglist

    def check_constraints(self) :
        self.gate_type.constraints(self)
        
    def copy(self) :
        obj = Gate(self.gate_type)
        obj.set_adjoint(self.adjoint)
        if self.ctrllist is not None :
            obj.set_ctrllist(list(self.ctrllist))
        obj.set_qreglist(list(self.qreglist))
        return obj

class ComposedGate(Operator) :
    def __init__(self, gate_type) :
        Operator.__init__(self)
        self.gate_type = gate_type
        self.adjoint = False
        self.gatelist = None
        self.ctrllist = None

    def set_adjoint(self, adjoint) :
        self.adjoint = adjoint
        
    def set_ctrllist(self, ctrllist) :
        assert self.ctrllist is None, 'ctrl args already set.'
        self.ctrllist = [ctrllist] if isinstance(ctrllist, Qreg) else ctrllist

    def set_gatelist(self, gatelist) :
        assert self.gatelist is None, 'gatelist already set.'
        assert isinstance(gatelist[0], Gate), 'Error'
        self.gatelist = gatelist

    def check_constraints(self) :
        self.gate_type.constraints(self)
    
    def copy(self) :
        obj = ComposedGate(self.gate_type)
        obj.set_adjoint(self.adjoint)
        if self.ctrllist is not None :
            obj.set_ctrllist(list(self.ctrllist))
        gatelist = []
        for gate in self.gatelist :
            gatelist.append(gate.copy())
        obj.set_gatelist(gatelist)
        return obj

    
# Currently only SWAP gate uses this.
class MultiQubitGate(Operator) :
    def __init__(self, gate_type) :
        self.gate_type = gate_type
        self.adjoint = False
        self.qreglist = None

    def set_adjoint(self, adjoint) :
        self.adjoint = adjoint

    def set_qreglist(self, qreglist) :
        assert self.qreglist is None, 'qreglist already set.'
        self.qreglist = qreglist

    def check_constraints(self) :
        self.gate_type.constraints(self)
    
    def copy(self) :
        obj = MultiQubitGate(self.gate_type)
        obj.set_adjoint(self.adjoint)
        obj.set_qreglist(list(self.qreglist))
        return obj
    
class Measure(Operator) :
    def __init__(self, ref, qreg) :
        if not isinstance(qreg, Qreg) or not isinstance(ref, Reference) :
            raise RuntimeError('Wrong argument for Measure, {}, {}.'.format(repr(qreg), repr(ref)))
        Operator.__init__(self)
        self.qreg, self.outref = qreg, ref
    
    def copy(self) :
        return Measure(self.outref, self.qreg)

class Pmeasure(Operator) :
    def __init__(self, ref, gatelist) :
        if not isinstance(ref, Reference) :
            raise RuntimeError('Wrong argument for Pmeasure, {}.'.format(repr(ref)))
        from . import gate_type as gtype 
        for gate in gatelist :
            if not isinstance(gate.gate_type, (gtype.ID, gtype.X, gtype.Y, gtype.Z)) :
                raise RuntimeError('exp gate only accepts ID, X, Y and Z gates')
            if len(gate.qreglist) != 1 :
                raise RuntimeError('# qregs must be 1.')
            if gate.ctrllist is not None :
                raise RuntimeError('control qreg(s) should not be set for pauli operators.')
        Operator.__init__(self)
        self.gatelist, self.outref = gatelist, ref
    
    def copy(self) :
        gatelist = [gate.copy() for gate in self.gatelist]
        return Pmeasure(self.outref, gatelist)
    
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
    
    def copy(self) :
        return Reset(self.qregset)
                
               
class Clause(Operator) :
    def __init__(self) :
        Operator.__init__(self)
        self.ops = []

    def add_op(self, op) :
        assert isinstance(op, Operator), "Unknown argument, {}.".format(repr(op))
        self.ops.append(op)

    def add(self, *args) :
        for arg in args :
            if isinstance(arg, (list, tuple, set)) :
                clause = Clause()
                clause.add(*arg)
                self.ops.append(clause)
            elif isinstance(arg, Operator) :
                self.ops.append(arg)
            else :
                assert False, 'Unknown argument, {}'.format(repr(arg))

class IfClause(Operator) :
    def __init__(self, refs, val) :
        Operator.__init__(self)
        self.refs = refs
        self.val = val

    def set_clause(self, clause) :
        self.clause = clause
