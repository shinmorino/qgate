import qgate.model as model
import qgate.model.gate_type as gtype

def cx(control, target) :
    g = model.Gate(gtype.X())
    g.set_ctrllist([control])
    g.set_qreglist([target])
    return g

def ca(control, target) :
    g = model.Gate(gtype.ID())
    g.set_ctrllist([control])
    g.set_qreglist([target])
    return g

# def swap(qreg0, qreg1) :
#    s = model.MultiQubitGate(gtype.SWAP())
#    s.set_qreglist([qreg0, qreg1])
#    return s

def swap(qreg0, qreg1) :
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

def expiI(theta, qreg0) :
    g = model.Gate(gtype.ExpiI(theta))
    g.set_qreglist([qreg0])
    return g

def expiZ(theta, qreg0) :
    g = model.Gate(gtype.ExpiZ(theta))
    g.set_qreglist([qreg0])
    return g

def a(qreglist) :
    s = model.Gate(gtype.ID())
    s.set_qreglist(qreglist)
    return s

def x(qreglist) :
    x = model.Gate(gtype.X())
    x.set_qreglist(qreglist)
    return x

def y(qreglist) :
    y = model.Gate(gtype.Y())
    y.set_qreglist(qreglist)
    return y

def z(qreglist) :
    z = model.Gate(gtype.Z())
    z.set_qreglist(qreglist)
    return z

def h(qreglist) :
    h = model.Gate(gtype.H())
    h.set_qreglist(qreglist)
    return h

def sh(qreglist) :
    sh = model.Gate(gtype.SH())
    sh.set_qreglist(qreglist)
    return sh
