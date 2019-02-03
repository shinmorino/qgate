import qgate.model as model
    
def new_circuit() :
    return model.Clause()

def allocate_qreg() :
    return model.Qreg()

def allocate_qregs(count) :
    return [model.Qreg() for _ in range(count)]

def allocate_creg() :
    return model.Creg()

def allocate_cregs(count) :
    return [model.Creg() for _ in range(count)]

# functions to instantiate operators

def measure(qreg, outref) :
    return model.Measure(qreg, outref)

def barrier(*qregs) :
    bar = model.Barrier(qregs)
    return bar

def reset(*qregs) :
    reset = model.Reset(qregs)
    return reset
        
def clause(*ops) :
    cl = model.Clause()
    for op in ops :
        cl.add_op(op)
    return cl

def if_(cregs, val, ops) :
    if isinstance(cregs, model.Creg) :
        cregs = [cregs]
    if_clause = model.IfClause(cregs, val)
    cl = clause(ops)
    if_clause.set_clause(cl)
    return if_clause


#
# Gate
#
import qgate.model.gate as gate

import sys
this = sys.modules[__name__]


# module level
#  gate type + gate parameters


class GateWrapper :
    def __init__(self, gate) :
        self.gate = gate

    def __call__(self, qreglist) :
        self.gate.set_qreglist(qreglist)
        return self.gate

class ConstGateFactory :
    def __init__(self, gate_type) :
        self.gate_type = gate_type

    def __call__(self, *qregs) :
        g = model.Gate(self.gate_type)
        g.set_qreglist(qregs)
        return g


# // idle gate (identity)
# gate id a { U(0,0,0) a; }
this.a = ConstGateFactory(gate.ID())

# // Clifford gate: Hadamard
# gate h a { u2(0,pi) a; }
this.h = ConstGateFactory(gate.H())

# // Clifford gate: sqrt(Z) phase gate
# gate s a { u1(pi/2) a; }
this.s = ConstGateFactory(gate.S())

# // C3 gate: sqrt(S) phase gate
# gate t a { u1(pi/4) a; }
this.t = ConstGateFactory(gate.T())

# // Pauli gate: bit-flip
# gate x a { u3(pi,0,pi) a; }
this.x = ConstGateFactory(gate.X())

# // Pauli gate: bit and phase flip
# gate y a { u3(pi,pi/2,pi/2) a; }
this.y = ConstGateFactory(gate.Y())

# // Pauli gate: phase flip
# gate z a { u1(pi) a; }
this.z = ConstGateFactory(gate.Z())

# // Rotation around X-axis
# gate rx(theta) a { u3(theta,-pi/2,pi/2) a; }
def rx(theta) :
    return GateWarpper(model.gate(gate.RX(theta)))

# // rotation around Y-axis
# gate ry(theta) a { u3(theta,0,0) a; }
def ry(theta) :
    return GateWarpper(model.gate(gate.RY(theta)))

# // rotation around Z axis
# gate rz(phi) a { u1(phi) a; }
def rz(theta) :
    return GateWarpper(model.gate(gate.RZ(theta)))

# 1 parameeter

# // 1-parameter 0-pulse single qubit gate
# gate u1(lambda) q { U(0,0,lambda) q; }
def u1(_lambda) :
    return GateWrapper(model.Gate(gate.U1(_lambda)))

# 2 parameeters

# // 2-parameter 1-pulse single qubit gate
# gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
def u2(phi, _lambda) :
    return GateWrapper(model.Gate(gate.U2(phi, _lambda)))

# 3 parameeters

# // --- QE Hardware primitives ---
# // 3-parameter 2-pulse single qubit gate
# gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
def u3(theta, phi, _lambda) :
    return GateWrapper(model.Gate(gate.U(theta, phi, _lambda)))


class ControlledGateFactory :

    def __init__(self, control) :
        # FIXME: add input check and expand args here.
        self.control = control
        
    def create(self, gtype) :
        g = model.Gate(gtype)
        g.set_control(self.control)
        return GateWrapper(g)
        
    @property
    def a(self) :
        return self.create(gate.ID())

    @property
    def h(self) :
        return self.create(gate.H())
    
    @property
    def s(self) :
        return self.create(gate.S())

    @property
    def t(self) :
        return self.create(gate.T())

    @property
    def x(self) :
        return self.create(gate.X())

    @property
    def y(self) :
        return self.create(gate.Y())

    @property
    def z(self) :
        return self.create(gate.Z())
    
    def rx(self, theta) :
        return self.create(gate.RX(theta))

    def ry(self, theta) :
        return self.create(gate.RY(theta))

    def rz(self, theta) :
        return self.create(gate.RZ(theta))
    
    def u1(self, _lambda) :
        return self.create(gate.U1(_lambda))

    def u2(self, phi, _lambda) :
        return self.create(gate.U2(phi, _lambda))

    def u3(self, theta, phi, _lambda) :
        return self.create(gate.U(theta, phi, _lambda))


def controlled(*control) :
    return ControlledGateFactory(control);

this.cntr = controlled


    
