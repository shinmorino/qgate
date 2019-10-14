from . import model
from . import gate_type as gtype
from .pauli_gates_diagonalizer import PauliGatesDiagonalizer
from .gate_factory import cx, expiI, expiZ

def adjoint(gates) :
    adj = list()
    for gate in reversed(gates) :
        copied = gate.copy()
        copied.adjoint = not gate.adjoint
        adj.append(copied)
    return adj

def expand_swap(op) :
    qreg0, qreg1 = op.qreglist[0], op.qreglist[1]
    return [cx(qreg0, qreg1), cx(qreg1, qreg0), cx(qreg0, qreg1)]

def expand_exp(exp) :
    diag = PauliGatesDiagonalizer(exp.gatelist)

    is_z_based = diag.diagonalize()
    pcx = diag.get_pcx()
    if diag.phase_offset_in_pi_2 % 2 != 0:
        raise RuntimeError('cannot expand, {}.'.format(repr(exp)))
    assert abs(diag.get_phase_coef()) == 1

    phase = diag.get_phase_coef() * exp.gate_type.args[0]
    if is_z_based :
        expgate = expiZ(phase.real, diag.op_qreg)
    else :
        expgate = expiI(phase.real, diag.op_qreg)

    expanded = pcx + [expgate] + adjoint(pcx)

    if exp.ctrllist is not None :
        for gate in expanded :
            gate.set_ctrllist(op.ctrllist + gate.ctrllist)
            # FIXME: remove later.
            assert len(op.ctrllist & gate.ctrllist) == 0, 'control bits must not overlap qregs.'

    if exp.adjoint :
        for gate in expgates :
            expgates.set_adjoint(True)
    else :
        expanded.reverse()

    return expanded

def expand_pmeasure(pmeasure) :
    diag = PauliGatesDiagonalizer(pmeasure.gatelist)

    if not diag.diagonalize() :
        raise RuntimeError('measurement is not z-based.')

    mop = model.Measure(pmeasure.outref, diag.op_qreg)
    pcx = diag.get_pcx()
    pcxadj = adjoint(pcx)
    expanded = pcx + [mop] + pcxadj
    expanded.reverse()
    return expanded

def expand_pprob(pprob) :
    diag = PauliGatesDiagonalizer(pprob.gatelist)

    if not diag.diagonalize() :
        raise RuntimeError('measurement is not z-based.')

    prob = model.Prob(pprob.outref, diag.op_qreg)
    pcx = diag.get_pcx()
    pcxadj = adjoint(pcx)
    expanded = pcx + [prob] + pcxadj
    expanded.reverse()
    return expanded

def expand(op) :
    # simply decompose to 3 cx gates, since runtimes does not have 2 qubit gate operations now.
    if isinstance(op, model.PauliMeasure) :
        return expand_pmeasure(op)
    elif isinstance(op, model.PauliProb) :
        return expand_pprob(op)
    elif isinstance(op.gate_type, gtype.SWAP) :
        return expand_swap(op)
    elif isinstance(op.gate_type, gtype.Expi) :
        return expand_exp(op)

    assert False, 'Unknown composed gate, {}.'.format(repr(op.gate_type))
