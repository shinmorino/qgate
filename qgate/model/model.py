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

    

def _expand_args(args) :
    expanded = []
    if isinstance(args, (list, tuple, set)) :
        for child in args :
            expanded += _expand_args(child)
    else :
        expanded.append(args)
    return expanded


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

    # matrix creator
    def get_matrix(self) :
        return self.__class__.mat(*self.args)
    
        
class Gate(Operator) :

    def __init__(self, gate_type) :
        Operator.__init__(self)
        self.gate_type = gate_type
        self.qreglist = None
        self.cntrlist = None

    # removed later
    def get_matrix(self) :
        return self.gate_type.get_matrix()

    def set_control(self, cntrlist) :
        assert self.cntrlist is None, 'cntr args already set.'
        self.cntrlist = [cntrlist] if isinstance(cntrlist, Qreg) else cntrlist

    def set_qreglist(self, qreglist) :
        assert self.qreglist is None, 'qreg list already set.'
        self.qreglist = _expand_args(qreglist)

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

    def add(self, *type_list) :
        self.type_list = type_list
        
    def set_control(self, cntrlist) :
        assert self.cntrlist is None, 'cntr args already set.'
        self.cntrlist = [cntrlist] if isinstance(cntrlist, Qreg) else cntrlist

    def set_qreglist(self, qreglist) :
        assert self.qreglist is None, 'qreg list already set.'
        self.qreglist = _expand_args(qreglist)

    def get_clause(self) :
        return None
        
    # FIXME: rename
    def create(self, qreglist, cntrlist) :
        obj = Gate(self.gate_type)
        obj.set_control(cntrlist)
        obj.set_qreglist(qreglist)
        return obj
    
    
class Measure(Operator) :
    def __init__(self, qreg, outref) :
        # FIXME: Better input check
        if not isinstance(qreg, Qreg) or not isinstance(outref, Creg) :
            raise RuntimeError('Wrong argument for Measure, {}, {}.'.format(repr(qreg), repr(outref)))
        Operator.__init__(self)
        self.qreg, self.outref = qreg, outref


class Barrier(Operator) :
    def __init__(self, *args) :
        Operator.__init__(self)
        args = _expand_args(args)
        assert all([isinstance(item, Qreg) for item in args]), 'arguments must be Qreg.'
        self.qregset = set(args)

class Reset(Operator) :
    def __init__(self, *args) :
        Operator.__init__(self)
        args = _expand_args(args)
        assert all([isinstance(item, Qreg) for item in args]), 'arguments must be Qreg.'
        self.qregset = set(args)
                
               
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
