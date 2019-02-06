# // Quantum Experience (QE) Standard Header
# // file: qelib1.inc
#

# FIXME: define role of this file

import math
import numpy as np
import qgate.model.gate as gate
from qgate.model.gate import adjoint
from .script import clause


# // controlled-NOT
# gate cx c,t { CX c,t; }
def cx(c, t) :
    return controlled.x(cntr)(t);

# // Clifford gate: conjugate of sqrt(Z)
# gate sdg a { u1(-pi/2) a; }

def sdg(a) :
    return s.H(a)

# // C3 gate: conjugate of sqrt(S)
# gate tdg a { u1(-pi/4) a; }
def tdg(a) :
    return t.H(a)

#// --- QE Standard User-Defined Gates  ---

# // controlled-Phase
# gate cz a,b { h b; cx a,b; h b; }
def cz(a, b) :
    return controlled.z(a)(b)

def _cz(a, b) :
    return clause(h(b), cx(a, b), h(b))

# // controlled-Y
# gate cy a,b { sdg b; cx a,b; s b; }

def cy(a ,b) :
    return controlled.y(a)(b)
    
def _cy(a, b) :
    return clause(sdg(b), cx(a, b), s(b))

# // controlled-H
# gate ch a,b {
# h b; sdg b;
# cx a,b;
# h b; t b;
# cx a,b;
# t b; h b; s b; x b; s a;
# }
def ch(a, b) :
    return controlled.H(a)(b)

def _ch(a, b) :
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
def crz(_lambda, a, b) :
    return controlled(a).rz(theta)(b)

def _crz(lambda_, a, b) :
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

def cu1(_lambda, a, b) :
    return controlled(a).u1(_lambda)(b)

def _cu1(lambda_, a, b) :
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
def cu3(theta, phi, _lambda, c, t) :
    return controlled(c).u3(theta, phi, _lambda)(t)

def _cu3(theta, phi, _lambda, c, t) :
    return clause(u1((lambda_ - phi) / 2., t),
                  cx(c, t),
                  u3(- theta / 2.0, - (phi + lambda_) / 2., t),
                  cx(c, t),
                  u3(theta / 2., phi, 0, t))
