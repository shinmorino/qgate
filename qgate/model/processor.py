from . import model
from . import gate


def update_clause_registers(clause) :
    qregs = set()
    cregs = set()
    for op in clause.ops :
        if isinstance(op, model.Measure) :
            qregs |= set(op.in0)
            cregs |= set(op.cregs)
        elif isinstance(op, gate.UnaryGate) :
            qregs |= set(op.in0)
        elif isinstance(op, gate.ControlGate) :
            qregs |= set(op.in0 + op.in1)
        elif isinstance(op, (model.Barrier, model.Reset)) :
            qregs |= op.qregset
        elif isinstance(op, model.Clause) :
            update_clause_registers(op)
            qregs |= op.get_qregset()
            cregs |= op.get_cregset()
        elif isinstance(op, model.IfClause) :
            update_clause_registers(op.clause)
            qregs |= op.clause.get_qregset()
            cregs |= op.clause.get_cregset()
        else :
            raise RuntimeError()
        
    clause.set_qregset(qregs)
    clause.set_cregset(cregs)


# merging qreg groups by checking qreg intersection
def _merge_qreg_groups(groups) :

    merged = []
    
    while len(groups) != 0 :
        unmerged = []
        
        group = groups[0]
        groups.remove(group)

        for other in groups :
            intersect = group & other
            if len(intersect) != 0 :
                group |= other
            else :
                unmerged.append(other)

        merged.append(group)
        groups = unmerged

    return merged



def isolate_qreg_groups(clause) :
    qreg_groups = []

    for op in clause.ops :
        if isinstance(op, model.ControlGate) :
            for in0, in1 in zip(op.in0, op.in1) :
                qreg_groups.append(set([in0, in1]))
        elif isinstance(op, model.Clause) :
            inner_qreg_groups = isolate_qreg_groups(op)
            qreg_groups += inner_qreg_groups
        elif isinstance(op, model.IfClause) :
            inner_qreg_groups = isolate_qreg_groups(op.clause)
            qreg_groups += inner_qreg_groups

    used_qregs = set()
    for group in qreg_groups :
        used_qregs |= group

    unused_qregs = clause.qregset - used_qregs
    for qreg in unused_qregs :
        qreg_groups.append(set([qreg]))

    qreg_groups = _merge_qreg_groups(qreg_groups)
        
    return qreg_groups


def _overlap_1(qregset, qregs) :
    extracted = []
    for qreg in qregs :
        if qreg in qregset :
            extracted.append(qreg)
    return extracted

def _overlap_2(qregset, qregs0, qregs1) :
    extracted0, extracted1 = [], []
    for qreg0, qreg1 in zip(qregs0, qregs1) :
        if qreg0 in qregset or qreg1 in qregset :
            extracted0.append(qreg0)
            extracted1.append(qreg1)

    return extracted0, extracted1
    
    
def _extract_cregs(qregset, qregs, cregs) :
    extracted = []
    for qreg, creg in zip(qregs, cregs) :
        if qreg in qregset :
            extracted.append(creg)
    return extracted


# clone operators if an operator belongs two or more circuits
def extract_operators(qregset, clause) :
    extracted = model.Clause()
    
    for op in clause.ops :
        new_op = None
        if isinstance(op, model.Measure) :
            in0 = _overlap_1(qregset, op.in0)
            cregs = _extract_cregs(in0, op.in0, op.cregs)
            if len(in0) != 0 :
                new_op = model.Measure(in0, cregs)
        elif isinstance(op, gate.U) :
            in0 = _overlap_1(qregset, op.in0)
            if len(in0) != 0 :
                new_op = op.__class__(op._theta, op._phi, op._lambda, in0)
        elif isinstance(op, gate.U2) :
            in0 = _overlap_1(qregset, op.in0)
            if len(in0) != 0 :
                new_op = op.__class__(op._phi, op._lambda, in0)
        elif isinstance(op, gate.U1) :
            in0 = _overlap_1(qregset, op.in0)
            if len(in0) != 0 :
                new_op = op.__class__(op._lambda, in0)
        elif isinstance(op, gate.UnaryGate) :
            in0 = _overlap_1(qregset, op.in0)
            if len(in0) != 0 :
                new_op = op.__class__(in0)
        elif isinstance(op, model.ControlGate) :
            control, target = _overlap_2(qregset, op.in0, op.in1)
            if len(control) != 0 :
                new_op = op.__class__(control, target)
                new_op.set_matrix(op.get_matrix())
        elif isinstance(op, model.Clause) :
            new_clause= extract_operators(qregset, op)
            if len(new_clause.ops) != 0 :
                new_op = new_clause
        elif isinstance(op, model.IfClause) :
            new_clause = extract_operators(qregset, op.clause)
            if len(new_clause.ops) != 0 :
                new_op = model.IfClause(op.creg_array, op.val)
                new_op.set_clause(new_clause)
        elif isinstance(op, (model.Barrier, model.Reset)) :
            qregs = _overlap_1(qregset, op.qregset)
            if len(qregs) != 0 :
                new_op = op.__class__(qregs)
        else :
            raise RuntimeError("Unknown operator")

        if new_op is not None :
            new_op.set_idx(op.get_idx())
            extracted.add_op(new_op)

    extracted.set_qregset(qregset)
        
    return extracted


# isolate clauses(circuits) from one clause.
def isolate_clauses(clause) :

    isolated = model.ClauseList()

    qreg_groups = isolate_qreg_groups(clause)

    used_qregs = set()
    for group in qreg_groups :
        used_qregs |= group
        
    # collect unused qregs
    unused_qregs = clause.qregset - used_qregs

    for qregs in qreg_groups :
        extracted = extract_operators(qregs, clause)
        isolated.append(extracted)

    # qregs with unary gates
    for qreg in unused_qregs :
        qregset = { qreg }
        extracted = extract_operators(qregset, clause)
        isolated.append(extracted)
        
    return isolated

    
def process(clause, **kwargs) :
    # get all registered qregs
    qregset_org = clause.get_qregset()
    cregset_org = clause.get_cregset()
    # update qregset for each clause including nested ones.
    update_clause_registers(clause)
    # get union to get all qregs.
    all_qregset = qregset_org | clause.qregset
    all_cregset = cregset_org | clause.cregset

    # give operators their numbers.
    for idx, op in enumerate(clause.ops) :
        op.set_idx(idx)
        
    if 'isolate_circuits' in kwargs.keys() and kwargs['isolate_circuits'] :
        isolated = isolate_clauses(clause)
    else :
        isolated = model.ClauseList()
        isolated.append(clause)

    # add all given qregs.
    unused_qregset = all_qregset - clause.qregset
    for qreg in unused_qregset :
        clause_1_qubit = model.Clause()
        clause_1_qubit.set_qregset({ qreg })
        isolated.append(clause_1_qubit)
        
    isolated.set_qregset(all_qregset)
    isolated.set_cregset(all_cregset)
            
    return isolated
