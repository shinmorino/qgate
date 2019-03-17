import qgate.model.model as model
import qgate.model.gate_type as gtype
from .composed_gate_decomposer import ComposedGateDecomposer
from .gate_factory import expiI, expiZ

def decompose_swap(op) :
    qreg0, qreg1 = op.qreglist[0], op.qreglist[1]
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

def decompose_exp(exp) :
    decomposer = ComposedGateDecomposer(exp)
    if not decomposer.filter_paulis() :
        # no pauli gates, return exp(i theta I)
        expgate = expiI(*exp.gate_type.args, exp.gatelist[0].qreglist)
        expgate.set_adjoint(exp.adjoint)
        return [expgate]

    if decomposer.decompose() :
        qreg = decomposer.op_qreg
        expgate = expiZ(*exp.gate_type.args, qreg)
    else :
        qreg = decomposer.op_qreg
        expgate = expiI(*exp.gate_type.args, qreg)

    decomposed = decomposer.get_pcx() + [expgate] + decomposer.get_pcxdg()
    return decomposed

def decompose(op) :
    # simply decompose to 3 cx gates, since runtimes does not have 2 qubit gate operations now.
    if isinstance(op.gate_type, gtype.SWAP) :
        return decompose_swap(op)
    elif isinstance(op.gate_type, gtype.Expi) :
        return decompose_exp(op)
    
    assert False, 'Unknown composed gate, {}.'.format(repr(op.gate_type))
