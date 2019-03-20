import qgate.model.model as model
import qgate.model.gate_type as gtype
from .composed_gate_decomposer import ComposedGateDecomposer
from .gate_factory import cx, expiI, expiZ

def decompose_swap(op) :
    qreg0, qreg1 = op.qreglist[0], op.qreglist[1]
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

def decompose_exp(exp) :
    decomposer = ComposedGateDecomposer(exp.gatelist)

    if decomposer.decompose() :
        expgate = expiZ(*exp.gate_type.args, decomposer.op_qreg)
    else :
        expgate = expiI(*exp.gate_type.args, decomposer.op_qreg)
        
    expgate.set_adjoint(exp.adjoint)
    decomposed = decomposer.get_pcx(exp.adjoint) + [expgate] + decomposer.get_pcxdg(exp.adjoint)
    
    if exp.ctrllist is not None :
        for gate in decomposed :
            gate.set_ctrllist(op.ctrllist + gate.ctrllist)
            # FIXME: remove later.
            assert len(op.ctrllist & gate.ctrllist) == 0, 'control bits must not overlap qregs.'
    
    return decomposed

def decompose_pmeasure(pmeasure) :
    # FIXME: id gates can be ignored.
    decomposer = ComposedGateDecomposer(pmeasure.gatelist)
    
    if not decomposer.decompose() :
        raise RuntimeError('not supported.')
    qreg = decomposer.op_qreg
    mop = model.Measure(pmeasure.outref, qreg)

    decomposed = decomposer.get_pcx(False) + [mop] + decomposer.get_pcxdg(False)
    return decomposed

def decompose_pprob(pprob) :
    # FIXME: id gates can be ignored.
    decomposer = ComposedGateDecomposer(pprob.gatelist)
    
    if not decomposer.decompose() :
        raise RuntimeError('not supported.')
    qreg = decomposer.op_qreg
    pop = model.Prob(pprob.outref, qreg)

    decomposed = decomposer.get_pcx(False) + [pop] + decomposer.get_pcxdg(False)
    return decomposed

def decompose(op) :
    # simply decompose to 3 cx gates, since runtimes does not have 2 qubit gate operations now.
    if isinstance(op, model.PauliMeasure) :
        return decompose_pmeasure(op)
    if isinstance(op, model.PauliProb) :
        return decompose_pprob(op)
    if isinstance(op.gate_type, gtype.SWAP) :
        return decompose_swap(op)
    elif isinstance(op.gate_type, gtype.Expi) :
        return decompose_exp(op)
    
    assert False, 'Unknown composed gate, {}.'.format(repr(op.gate_type))
