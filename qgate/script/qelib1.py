# // Quantum Experience (QE) Standard Header
# // file: qelib1.inc

from .script import S, T, ctrl

# // controlled-NOT
def cx(c, t) :
    return ctrl(c).X(t);

# // Clifford gate: conjugate of sqrt(Z)
def sdg(a) :
    return S.Adj(a)

# // C3 gate: conjugate of sqrt(S)
def tdg(a) :
    return T.Adj(a)

#// --- QE Standard User-Defined Gates  ---

# // controlled-Phase
def cz(a, b) :
    return ctrl(a).Z(b)

# // controlled-Y
def cy(a ,b) :
    return ctrl(a).Y(b)

# // controlled-H
def ch(a, b) :
    return ctrl(a).H(b)

# // C3 gate: Toffoli
def ccx(a, b, c) :
    return ctrl(a, b).X(c)

# // controlled rz rotation
def crz(_lambda, a, b) :
    return ctrl(a).Rz(_lambda)(b)

# // controlled phase rotation
def cu1(_lambda, a, b) :
    return ctrl(a).U1(_lambda)(b)

# // controlled-U
def cu3(theta, phi, _lambda, c, t) :
    return ctrl(c).U3(theta, phi, _lambda)(t)
