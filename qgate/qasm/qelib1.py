# // Quantum Experience (QE) Standard Header
# // file: qelib1.inc
#

import math
import numpy as np
import qgate.model.model as model
from qgate.model.model import adjoint


# // --- QE Hardware primitives ---
# // 3-parameter 2-pulse single qubit gate
# gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
def u3(theta, phi, lambda_, q) :
    return model.U(theta, phi, lambda_, q)

# // 2-parameter 1-pulse single qubit gate
# gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
def u2(phi, lambda_, q) :
    return model.U2(phi, lambda_, q)

# // 1-parameter 0-pulse single qubit gate
# gate u1(lambda) q { U(0,0,lambda) q; }
def u1(lambda_, q) :
    return model.U1(lambda_, q)

# // controlled-NOT
# gate cx c,t { CX c,t; }
def cx(c, t) :    
    return model.CX(c, t)

# // idle gate (identity)
# gate id a { U(0,0,0) a; }

def a(a) :
    return model.ID(a)

# // --- QE Standard Gates ---
#
# // Pauli gate: bit-flip
# gate x a { u3(pi,0,pi) a; }

def x(a) :
    return model.X(a)

# // Pauli gate: bit and phase flip
# gate y a { u3(pi,pi/2,pi/2) a; }

def y(a) :
    return model.Y()

# // Pauli gate: phase flip
# gate z a { u1(pi) a; }

def z(a) :
    return model.Z(a)

# // Clifford gate: Hadamard
# gate h a { u2(0,pi) a; }

def h(a) :   
    return model.H(a)

# // Clifford gate: sqrt(Z) phase gate
# gate s a { u1(pi/2) a; }
def s(a) :
    return model.S(a)

# // Clifford gate: conjugate of sqrt(Z)
# gate sdg a { u1(-pi/2) a; }

class SDG(model.UnaryGate) : # Conjugate of S
    # gate sdg a { u1(-pi/2) a; }
    mat = adjoint(model.S.mat)
    def __init__(self, qregs) :
        model.UnaryGate.__init__(self, qregs)
        self.set_matrix(SDG.mat)

def sdg(a) :
    return SDG(a)

# // C3 gate: sqrt(S) phase gate
# gate t a { u1(pi/4) a; }

def t(a) :
    return model.T(a)

# // C3 gate: conjugate of sqrt(S)
# gate tdg a { u1(-pi/4) a; }

class TDG(model.UnaryGate) :  # T in Q#
    # gate tdg a { u1(-pi/4) a; }
    mat = adjoint(model.T.mat)
    def __init__(self, qregs) :
        UnaryGate.__init__(self, qregs)
        self.set_matrix(TDG.mat)

def tdg(a) :
    return TGD(a)

# // --- Standard rotations ---
# // Rotation around X-axis
# gate rx(theta) a { u3(theta,-pi/2,pi/2) a; }

def rx(theta, a) :
    return model.RX(theta, a)

# // rotation around Y-axis
# gate ry(theta) a { u3(theta,0,0) a; }

def ry(theta, a) :
    return model.RY(theta, a)

# // rotation around Z axis
# gate rz(phi) a { u1(phi) a; }

def rz(phi, a) :
    return model.RZ(phi, a)

#// --- QE Standard User-Defined Gates  ---

# // controlled-Phase
# gate cz a,b { h b; cx a,b; h b; }

def cz(a, b) :
    return model.clause(h(b), cx(a, b), h(b))

# // controlled-Y
# gate cy a,b { sdg b; cx a,b; s b; }

def cy(a, b) :
    return model.clause(sdg(b), cx(a, b), s(b))


# // controlled-H
# gate ch a,b {
# h b; sdg b;
# cx a,b;
# h b; t b;
# cx a,b;
# t b; h b; s b; x b; s a;
# }

def ch(a, b) :
    return clause(h(b), sdg(b), cx(a,b),
                  h(b), t(b),
                  cx(a, b),
                  t(b), h(b), s(b), x(b), s(a))

# // C3 gate: Toffoli
# gate ccx a,b,c
# {
#   h c;
#   cx b,c; tdg c;
#   cx a,c; t c;
#   cx b,c; tdg c;
#   cx a,c; t b; t c; h c;
#   cx a,b; t a; tdg b;
#   cx a,b;
# }
def ccx(a, b, c) :
    return clause(cx(b, c), tdg(c),
                  cx(a, c), t(c),
                  cx(b, c), tdg(c),
                  cx(a, c), t(b), t(c), h(c),
                  cx(a, b), t(a), tdg(b),
                  cx(a, b))

# // controlled rz rotation
# gate crz(lambda) a,b
# {
#  u1(lambda/2) b;
#  cx a,b;
#  u1(-lambda/2) b;
#  cx a,b;
# }
def crz(lambda_, a, b) :
    return clause(u1(lambda_/2., b),
                  cx(a, b),
                  u1(- lambda_ / 2., b),
                  cx(a, b))

# // controlled phase rotation
# gate cu1(lambda) a,b
# {
#   u1(lambda/2) a;
#   cx a,b;
#   u1(-lambda/2) b;
#   cx a,b;
#   u1(lambda/2) b;
# }

def cu1(lambda_, a, b) :
    return clause(u1(lambda_ / 2., a),
                  cx(a, b),
                  u1(- lambda_ / 2., b),
                  cx(a, b),
                  u1(lambda_ / 2., b))

# // controlled-U
# gate cu3(theta,phi,lambda) c, t
# {
#   // implements controlled-U(theta,phi,lambda) with  target t and control c
#   u1((lambda-phi)/2) t;
#   cx c,t;
#   u3(-theta/2,0,-(phi+lambda)/2) t;
#   cx c,t;
#   u3(theta/2,phi,0) t;
# }

def cu3(theta, phi, lambda_, c, t) :
    return clause(u1((lambda_ - phi) / 2., t),
                  cx(c, t),
                  u3(- theta / 2.0, - (phi + lambda_) / 2., t),
                  cx(c, t),
                  u3(theta / 2., phi, 0, t))
