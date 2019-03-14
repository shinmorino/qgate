import qgate.model.model as model
import qgate.model.gate_type as gtype

def cx(control, target) :
    g = model.Gate(gtype.X())
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

def decompose(op) :
    # simply decompose to 3 cx gates, since runtimes does not have 2 qubit gate operations now.
    if isinstance(op.gate_type, gtype.SWAP) :
        return decompose_swap(op)
    
    assert False, 'Unknown composed gate, {}.'.format(repr(op.gate_type))
