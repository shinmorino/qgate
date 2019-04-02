from . import model
from . import gate_type as gtype
from .composed_gate_decomposer import ComposedGateDecomposer
from .gate_factory import cx, expiI, expiZ

def decompose_swap(op) :
    qreg0, qreg1 = op.qreglist[0], op.qreglist[1]
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

def decompose_exp(exp) :
    decomposer = ComposedGateDecomposer(exp.gatelist)

    is_z_based = decomposer.decompose()
    phase = exp.gate_type.args[0] + decomposer.get_phase_offset()
    
    if  is_z_based :
        expgate = expiZ(phase, decomposer.op_qreg)
    else :
        expgate = expiI(phase, decomposer.op_qreg)
        
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

    phase = decomposer.get_phase_offset()
    exp = expiZ(phase, decomposer.op_qreg)
    exp_adj = expiZ(-phase, decomposer.op_qreg)
    qreg = decomposer.op_qreg
    mop = [exp, model.Measure(pmeasure.outref, qreg), exp_adj]
    decomposed = decomposer.get_pcx(False) + mop + decomposer.get_pcxdg(False)
    return decomposed

def decompose_pprob(pprob) :
    # FIXME: id gates can be ignored.
    decomposer = ComposedGateDecomposer(pprob.gatelist)
    
    if not decomposer.decompose() :
        raise RuntimeError('not supported.')
    phase = decomposer.get_phase_offset()
    exp = expiZ(phase, decomposer.op_qreg)
    exp_adj = expiZ(-phase, decomposer.op_qreg)
    qreg = decomposer.op_qreg
    pop = [exp, model.Prob(pprob.outref, qreg), exp_adj]

    decomposed = decomposer.get_pcx(False) + pop + decomposer.get_pcxdg(False)
    return decomposed

def decompose(op) :
    # simply decompose to 3 cx gates, since runtimes does not have 2 qubit gate operations now.
    if isinstance(op, model.PauliMeasure) :
        return decompose_pmeasure(op)
    elif isinstance(op, model.PauliProb) :
        return decompose_pprob(op)
    elif isinstance(op.gate_type, gtype.SWAP) :
        return decompose_swap(op)
    elif isinstance(op.gate_type, gtype.Expi) :
        return decompose_exp(op)
    
    assert False, 'Unknown composed gate, {}.'.format(repr(op.gate_type))
