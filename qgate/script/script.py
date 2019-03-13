import qgate.model as model
import numbers

def _expand_args(args) :
    expanded = []
    if isinstance(args, (list, tuple, set)) :
        for child in args :
            expanded += _expand_args(child)
    else :
        expanded.append(args)
    return expanded

def _assert_is_number(obj) :
    if not isinstance(obj, numbers.Number) :
        raise RuntimeError('{} is not a number'.format(str(obj)))

def new_circuit() :
    return model.Clause()

def new_qreg() :
    return model.Qreg()

def new_qregs(count) :
    return [model.Qreg() for _ in range(count)]

def new_reference() :
    return model.Reference()

def new_references(count) :
    return [model.Reference() for _ in range(count)]

# functions to instantiate operators

def measure(qreg, outref) :
    return model.Measure(qreg, outref)

def barrier(*qregs) :
    qregs = _expand_args(qregs)
    bar = model.Barrier(qregs)
    return bar

def reset(*qregs) :
    qregs = _expand_args(qregs)
    reset = model.Reset(qregs)
    return reset

def clause(*ops) :
    cl = model.Clause()
    cl.add(ops)
    return cl

def if_(refs, val, ops) :
    refs = _expand_args(refs)
    if_clause = model.IfClause(refs, val)
    cl = clause(ops)
    if_clause.set_clause(cl)
    return if_clause


#
# Gate
#
import qgate.model.gate_type as gtype

import sys
this = sys.modules[__name__]


# module level
#  gate type + gate parameters


class GateFactory :
    def __init__(self, gate_type) :
        self.gate = model.Gate(gate_type)

    @property
    def H(self) :
        self.gate.set_adjoint(True)
        return self

    def __call__(self, *qregs) :
        qreglist = _expand_args(qregs)
        self.gate.set_qreglist(qreglist)
        self.gate.check_constraints()
        return self.gate

class ConstGateFactory :
    def __init__(self, gate_type) :
        self.gate_type = gate_type
        
    @property
    def H(self) :
        factory = GateFactory(self.gate_type)
        return factory.H
        
    def __call__(self, *qregs) :
        g = model.Gate(self.gate_type)
        qreglist = _expand_args(qregs)
        g.set_qreglist(qreglist)
        g.check_constraints()
        return g


# // idle gate (identity)
# gate id a { U(0,0,0) a; }
this.a = ConstGateFactory(gtype.ID())

# // Clifford gate: Hadamard
# gate h a { u2(0,pi) a; }
this.h = ConstGateFactory(gtype.H())

# // Clifford gate: sqrt(Z) phase gate
# gate s a { u1(pi/2) a; }
this.s = ConstGateFactory(gtype.S())

# // C3 gate: sqrt(S) phase gate
# gate t a { u1(pi/4) a; }
this.t = ConstGateFactory(gtype.T())

# // Pauli gate: bit-flip
# gate x a { u3(pi,0,pi) a; }
this.x = ConstGateFactory(gtype.X())

# // Pauli gate: bit and phase flip
# gate y a { u3(pi,pi/2,pi/2) a; }
this.y = ConstGateFactory(gtype.Y())

# // Pauli gate: phase flip
# gate z a { u1(pi) a; }
this.z = ConstGateFactory(gtype.Z())

# // Rotation around X-axis
# gate rx(theta) a { u3(theta,-pi/2,pi/2) a; }
def rx(theta) :
    _assert_is_number(theta)
    return GateFactory(gtype.RX(theta))

# // rotation around Y-axis
# gate ry(theta) a { u3(theta,0,0) a; }
def ry(theta) :
    _assert_is_number(theta)
    return GateFactory(gtype.RY(theta))

# // rotation around Z axis
# gate rz(phi) a { u1(phi) a; }
def rz(theta) :
    _assert_is_number(theta)
    return GateFactory(gtype.RZ(theta))

# 1 parameeter

# // 1-parameter 0-pulse single qubit gate
# gate u1(lambda) q { U(0,0,lambda) q; }
def u1(_lambda) :
    _assert_is_number(_lambda)
    return GateFactory(gtype.U1(_lambda))

# 2 parameeters

# // 2-parameter 1-pulse single qubit gate
# gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
def u2(phi, _lambda) :
    _assert_is_number(_lambda)
    return GateFactory(gtype.U2(phi, _lambda))

# 3 parameeters

# // --- QE Hardware primitives ---
# // 3-parameter 2-pulse single qubit gate
# gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
def u3(theta, phi, _lambda) :
    _assert_is_number(theta)
    _assert_is_number(phi)
    _assert_is_number(_lambda)
    return GateFactory(gtype.U(theta, phi, _lambda))

# exp
def expia(theta) :
    _assert_is_number(theta)
    return GateFactory(gtype.ExpiI(theta))

def expiz(theta) :
    _assert_is_number(theta)
    return GateFactory(gtype.ExpiZ(theta))

# utility
this.sh = ConstGateFactory(gtype.SH())


class ControlledGateFactory :

    def __init__(self, control) :
        self.control = _expand_args(control)
        
    def create(self, gtype) :
        factory = GateFactory(gtype)
        factory.gate.set_cntrlist(self.control)
        return factory
        
    @property
    def a(self) :
        return self.create(gtype.ID())

    @property
    def h(self) :
        return self.create(gtype.H())
    
    @property
    def s(self) :
        return self.create(gtype.S())

    @property
    def t(self) :
        return self.create(gtype.T())

    @property
    def x(self) :
        return self.create(gtype.X())

    @property
    def y(self) :
        return self.create(gtype.Y())

    @property
    def z(self) :
        return self.create(gtype.Z())
    
    def rx(self, theta) :
        _assert_is_number(theta)
        return self.create(gtype.RX(theta))

    def ry(self, theta) :
        _assert_is_number(theta)
        return self.create(gtype.RY(theta))

    def rz(self, theta) :
        _assert_is_number(theta)
        return self.create(gtype.RZ(theta))
    
    def u1(self, _lambda) :
        _assert_is_number(_lambda)
        return self.create(gtype.U1(_lambda))

    def u2(self, phi, _lambda) :
        _assert_is_number(phi)
        _assert_is_number(_lambda)
        return self.create(gtype.U2(phi, _lambda))

    def u3(self, theta, phi, _lambda) :
        _assert_is_number(theta)
        _assert_is_number(phi)
        _assert_is_number(_lambda)
        return self.create(gtype.U(theta, phi, _lambda))

    
    def expia(self, theta) :
        return self.create(gtype.ExpiI(theta))
    
    def expiz(self, theta) :
        return self.create(gtype.ExpiZ(theta))

    # utility
    @property
    def sh(self) :
        return self.create(gtype.SH())
    

def controlled(*control) :
    return ControlledGateFactory(control);

this.cntr = controlled


    
