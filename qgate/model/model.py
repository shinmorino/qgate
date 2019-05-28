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
        pass

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
        self.qreg = None
        self.ctrllist = None

    def set_adjoint(self, adjoint) :
        self.adjoint = adjoint

    def set_ctrllist(self, ctrllist) :
        self.ctrllist = [ctrllist] if isinstance(ctrllist, Qreg) else ctrllist
        assert all([isinstance(qreg, Qreg) for qreg in self.ctrllist]), 'arguments must be Qreg.'
        
    def set_qreg(self, qreg) :
        if not isinstance(qreg, Qreg) :
            raise RuntimeError('{} is not a qreg'.format(repr(qreg)))
        self.qreg = qreg

    def check_constraints(self) :
        self.gate_type.constraints(self)
        
    def copy(self) :
        obj = Gate(self.gate_type)
        obj.set_adjoint(self.adjoint)
        if self.ctrllist is not None :
            obj.set_ctrllist(list(self.ctrllist))
        obj.set_qreg(self.qreg)
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
    
class Prob(Operator) :
    def __init__(self, ref, qreg) :
        if not isinstance(qreg, Qreg) or not isinstance(ref, Reference) :
            raise RuntimeError('Wrong argument for Prob, {}, {}.'.format(repr(qreg), repr(ref)))
        Operator.__init__(self)
        self.qreg, self.outref = qreg, ref
    
    def copy(self) :
        return Prob(self.outref, self.qreg)

class PauliObserver(Operator) :
    def __init__(self, ref, gatelist) :
        if not isinstance(ref, Reference) :
            raise RuntimeError('Wrong argument for Pmeasure, {}.'.format(repr(ref)))
        from . import gate_type as gtype 
        for gate in gatelist :
            if not isinstance(gate.gate_type, (gtype.ID, gtype.X, gtype.Y, gtype.Z)) :
                raise RuntimeError('Pmeasure only accepts ID, X, Y and Z gates')
            if gate.ctrllist is not None :
                raise RuntimeError('control qreg(s) should not be set for pauli operators.')
        Operator.__init__(self)
        self.gatelist, self.outref = gatelist, ref

class PauliMeasure(PauliObserver) :
    def __init__(self, ref, gatelist) :
        PauliObserver.__init__(self, ref, gatelist)
    
    def copy(self) :
        gatelist = [gate.copy() for gate in self.gatelist]
        return PauliMeasure(self.outref, gatelist)

class PauliProb(PauliObserver) :
    def __init__(self, ref, gatelist) :
        PauliObserver.__init__(self, ref, gatelist)
    
    def copy(self) :
        gatelist = [gate.copy() for gate in self.gatelist]
        return PauliProb(self.outref, gatelist)
    
    
class Barrier(Operator) :
    def __init__(self, qregset) :
        Operator.__init__(self)
        assert all([isinstance(qreg, Qreg) for qreg in qregset]), 'arguments must be Qreg.'
        self.qregset = set(qregset)

    def copy(self) :
        return Barrier(self.qregset)
    

class Reset(Operator) :
    def __init__(self, qreg) :
        Operator.__init__(self)
        assert isinstance(qreg, Qreg), 'arguments must be Qreg.'
        self.qreg = qreg
    
    def copy(self) :
        return Reset(self.qreg)


class IfClause(Operator) :
    def __init__(self, refs, cond, clause) :
        Operator.__init__(self)
        self.refs = refs
        self.cond = cond
        self.clause = clause

    def copy(self) :
        return IfClause(self.refs, self.cond, self.clause)

