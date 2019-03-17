import qgate.model.model as model
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

def decompose_swap(op) :
    qreg0, qreg1 = op.qreglist[0], op.qreglist[1]
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

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

def expiI(theta, qreg0) :
    g = model.Gate(gtype.ExpiI(theta))
    g.set_qreglist([qreg0])
    return g

def expiZ(theta, qreg0) :
    g = model.Gate(gtype.ExpiZ(theta))
    g.set_qreglist([qreg0])
    return g

def _isID(op) :
    return isinstance(op.gate_type, gtype.ID)

def decompose_exp(exp) :
    # id gates are ignored.
    paulis = []
    for gate in exp.gatelist :
        if not _isID(gate) : # extract pauli gates
            paulis.append(gate)

    if len(paulis) == 0 :
        # no pauli gates, return exp(i theta I)
        expgate = expiI(*exp.gate_type.args, exp.gatelist[0].qreglist)
        expgate.set_adjoint(exp.adjoint)
        return [expgate]

    # collect gates according to qregs.
    gmap = dict()
    for gate in paulis :
        qreg = gate.qreglist[0]
        glist = gmap.get(qreg, None)
        if glist is None :
           gmap[qreg] = [gate]
        else :
            glist.append(gate)

    glist_even, glist_odd = [], []
    # decompose gmap for qregs 
    for glist in gmap.values() :
        if len(glist) % 2 == 0 :
            glist_even += glist
        else :
            glist_odd += glist
    
    plist = []
    for gate in glist_even + glist_odd :
        if isinstance(gate.gate_type, gtype.X) :
            # X = H Z H
            plist.append(h(gate.qreglist))
        elif isinstance(gate.gate_type, gtype.Y) :
            # Y = (SH) Z (HS+)
            plist.append(sh(gate.qreglist))
        elif isinstance(gate.gate_type, gtype.Z) :
            pass
        else :
            raise RuntimeError('input must be X, Y, Z or ID, but {} passed.'.format(gate.gate_type))
        
    # create adjoint P
    palist = [p.copy() for p in plist]
    for pa in palist :
        pa.set_adjoint(True)

    cxlist = []
    for i in range(len(glist_even) - 1) :
        d0, d1 = glist_even[i].qreglist[0], glist_even[i + 1].qreglist[0]
        cxlist.append(ca(d0, d1))

    if len(glist_odd) == 0 :
        expgate = expiI(*exp.gate_type.args, *glist_even[0].qreglist)
    else :
        if len(glist_even) != 0 :
            d0, d1 = glist_even[-1].qreglist[0], glist_odd[0].qreglist[0]
            cxlist.append(ca(d0, d1))
        for i in range(len(glist_odd) - 1) :
            d0, d1 = glist_odd[i].qreglist[0], glist_odd[i + 1].qreglist[0]
            cxlist.append(cx(d0, d1))
        expgate = expiZ(*exp.gate_type.args, *glist_odd[-1].qreglist)
            
    cxalist = reversed(cxlist)
    if exp.adjoint :
        # if adjoint, reverse order of cxlist.
        cxlist, cxalist = cxalist, cxlist
    expgate.set_adjoint(exp.adjoint)
        
    # reconstruct
    decomposed = plist
    decomposed += cxlist
    decomposed.append(expgate)
    decomposed += cxalist
    decomposed += palist

    if exp.ctrllist is not None :
        for gate in decomposed :
            gate.set_ctrllist(list(exp.ctrllist))
    
    return decomposed


def decompose(op) :
    # simply decompose to 3 cx gates, since runtimes does not have 2 qubit gate operations now.
    if isinstance(op.gate_type, gtype.SWAP) :
        return decompose_swap(op)
    elif isinstance(op.gate_type, gtype.Expi) :
        return decompose_exp(op)
    
    assert False, 'Unknown composed gate, {}.'.format(repr(op.gate_type))
