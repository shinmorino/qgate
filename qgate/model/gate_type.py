import numpy as np
import math
import cmath


def _1_bit_gate_constraints(self, gate) :
    if len(gate.qreglist) != 1 :
        raise RuntimeError('# qregs must be 1.')
    
    if gate.cntrlist is not None :
        if gate.qreglist[0] in gate.cntrlist :
            raise RuntimeError('control and operand overlapped.')
    
    
def _attach(gate_type, constraints) :
    gate_type.constraints = constraints
    

#
# built-in gate types
#

from .model import GateType

class U(GateType) :
    def __init__(self, theta, phi, _lambda) :
        GateType.__init__(self, theta, phi, _lambda);
_attach(U, _1_bit_gate_constraints)
            
class U2(GateType) :
    # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    def __init__(self, phi, _lambda) :
        GateType.__init__(self, phi,_lambda)
_attach(U2, _1_bit_gate_constraints)
            
class U1(GateType) :
    # gate u1(lambda) q { U(0,0,lambda) q; }
    def __init__(self, _lambda) :
        GateType.__init__(self, _lambda)
_attach(U1, _1_bit_gate_constraints)
        
class ID(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(ID, _1_bit_gate_constraints)

class X(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(X, _1_bit_gate_constraints)

class Y(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(Y, _1_bit_gate_constraints)

class Z(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(Z, _1_bit_gate_constraints)

class H(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(H, _1_bit_gate_constraints)

class S(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(S, _1_bit_gate_constraints)

class T(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(T, _1_bit_gate_constraints)

class RX(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(RX, _1_bit_gate_constraints)
        
class RY(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(RY, _1_bit_gate_constraints)

# utility gate to convert X to Z-basis.
class HSdg(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(HSdg, _1_bit_gate_constraints)


# RZ is an alias to U1
RZ = U1

# EXP
class ExpiI :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(ExpiI, _1_bit_gate_constraints)

class ExpiZ :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(ExpiZ, _1_bit_gate_constraints)

