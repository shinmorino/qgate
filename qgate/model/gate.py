import numpy as np
import math
import cmath


def adjoint(mat) :
    return np.conjugate(mat)

#
# builtin gate
#

#
# built-in gate implementation
#

from .model import GateType

class U(GateType) :    
    def __init__(self, theta, phi, _lambda) :
        GateType.__init__(self, theta, phi, _lambda);
            
class U2(GateType) :
    # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    def __init__(self, phi, _lambda) :
        GateType.__init__(self, phi,_lambda)
            
class U1(GateType) :
    # gate u1(lambda) q { U(0,0,lambda) q; }
    def __init__(self, _lambda) :
        GateType.__init__(self, _lambda)
        
class ID(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class X(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class Y(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class Z(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class H(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class S(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class T(GateType) :
    def __init__(self) :
        GateType.__init__(self)

class RX(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)
        
class RY(GateType) :
    def __init__(self, theta) :
        GateType.__init__(self, theta)

# RZ is an alias to U1
RZ = U1
