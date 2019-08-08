from . import model
from . import gate_type as gtype

def cx(control, target) :
    g = model.Gate(gtype.X())
    g.set_ctrllist([control])
    g.set_qreg(target)
    return g

def ci(control, target) :
    g = model.Gate(gtype.ID())
    g.set_ctrllist([control])
    g.set_qreg(target)
    return g

# def swap(qreg0, qreg1) :
#    s = model.MultiQubitGate(gtype.SWAP())
#    s.set_qreglist([qreg0, qreg1])
#    return s

def swap(qreg0, qreg1) :
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

def expiI(theta, qreg0) :
    g = model.Gate(gtype.ExpiI(theta))
    g.set_qreg(qreg0)
    return g

def expiZ(theta, qreg0) :
    g = model.Gate(gtype.ExpiZ(theta))
    g.set_qreg(qreg0)
    return g

def a(qreg) :
    s = model.Gate(gtype.ID())
    s.set_qreg(qreg)
    return s

def x(qreg) :
    x = model.Gate(gtype.X())
    x.set_qreg(qreg)
    return x

def y(qreg) :
    y = model.Gate(gtype.Y())
    y.set_qreg(qreg)
    return y

def z(qreg) :
    z = model.Gate(gtype.Z())
    z.set_qreg(qreg)
    return z

def h(qreg) :
    h = model.Gate(gtype.H())
    h.set_qreg(qreg)
    return h

def sh(qreg) :
    sh = model.Gate(gtype.SH())
    sh.set_qreg(qreg)
    return sh

def s(qreg) :
    s = model.Gate(gtype.S())
    s.set_qreg(qreg)
    return s
