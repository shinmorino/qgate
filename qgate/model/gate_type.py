import numpy as np
import math
import cmath


def _1_bit_gate_constraints(self, gate) :
    if gate.ctrllist is not None :
        if gate.qreg in gate.ctrllist :
            raise RuntimeError('control and operand overlapped.')
    

def _exp_gate_constraints(self, exp) :
    in_qregset = set()
    for gate in exp.gatelist :
        if not isinstance(gate.gate_type, (ID, X, Y, Z)) :
            raise RuntimeError('exp gate only accepts ID, X, Y and Z gates')
        if gate.ctrllist is not None :
            raise RuntimeError('control qreg(s) should not be set for exp gate parameters.')
        in_qregset.add(gate.qreg)
    
    if exp.ctrllist is not None :
        if len(in_qregset & set(exp.ctrllist)) != 0 :
            raise RuntimeError('control bit and target should not overlap.')
    
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

class RZ(GateType) :
    def __init__(self, _lambda) :
        GateType.__init__(self, _lambda)
_attach(RZ, _1_bit_gate_constraints)

# utility gate to convert X to Z-basis.
class SH(GateType) :
    def __init__(self) :
        GateType.__init__(self)
_attach(SH, _1_bit_gate_constraints)


# EXP
class ExpiI(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(ExpiI, _1_bit_gate_constraints)

class ExpiZ(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(ExpiZ, _1_bit_gate_constraints)

# composed gate

class Expi(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
_attach(Expi, _exp_gate_constraints)

# swap
class SWAP(GateType) :
    def __init__(self) :
        GateType.__init__(self)
