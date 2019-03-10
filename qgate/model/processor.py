from . import model
from .pseudo_operator import FrameBegin, FrameEnd

def update_clause_registers(clause) :
    qregs = set()
    refs = set()
    for op in clause.ops :
        if isinstance(op, model.Measure) :
            qregs.add(op.qreg)
            refs.add(op.outref)
        elif isinstance(op, model.Gate) :
            qregs |= set(op.qreglist)
            if not op.cntrlist is None :
                qregs |= set(op.cntrlist)
        elif isinstance(op, (model.Barrier, model.Reset)) :
            qregs |= op.qregset
        elif isinstance(op, model.Clause) :
            update_clause_registers(op)
            qregs |= op.get_qregset()
            refs |= op.get_refset()
        elif isinstance(op, model.IfClause) :
            update_clause_registers(op.clause)
            qregs |= op.clause.get_qregset()
            refs |= op.clause.get_refset()
        elif isinstance(op, (FrameBegin, FrameEnd)) :
            pass
        else :
            raise RuntimeError(repr(op))
        
    clause.set_qregset(qregs)
    clause.set_refset(refs)


# merging qreg groups by checking qreg intersection
def _merge_qreg_groups(groups) :

    n_groups = 0
    while len(groups) != n_groups :
        n_groups = len(groups)
        if n_groups == 1 :
            return groups

        processed = []
        while (len(groups) != 0)  :
            group = groups[0]
            groups.remove(group)
            
            overlapped = [other for other in groups if len(group & other) != 0]
            for frag in overlapped :
                group |= frag
                groups.remove(frag)

            # merged groups
            processed.append(group)
            
        groups = processed
        

    return groups



def isolate_qreg_groups(clause) :
    qreg_groups = []

    for op in clause.ops :
        if isinstance(op, model.Gate) :
            group = set(op.qreglist)
            if op.cntrlist is not None :
                group |= set(op.cntrlist)
            qreg_groups.append(group)
        elif isinstance(op, model.Clause) :
            inner_qreg_groups = isolate_qreg_groups(op)
            qreg_groups += inner_qreg_groups
        elif isinstance(op, model.IfClause) :
            inner_qreg_groups = isolate_qreg_groups(op.clause)
            qreg_groups += inner_qreg_groups

    qreg_groups = _merge_qreg_groups(qreg_groups)
    # print('qreg groups', len(qreg_groups))
            
    used_qregs = set()
    for group in qreg_groups :
        used_qregs |= group

    unused_qregs = clause.qregset - used_qregs
    for qreg in unused_qregs :
        qreg_groups.append(set([qreg]))
        
    return qreg_groups


def _overlap_1(qregset, qregs) :
    extracted = []
    for qreg in qregs :
        if qreg in qregset :
            extracted.append(qreg)
    return extracted


def _extract_refs(qregset, qregs, refs) :
    extracted = []
    for qreg, ref in zip(qregs, refs) :
        if qreg in qregset :
            extracted.append(ref)
    return extracted


# clone operators if an operator belongs two or more circuits
def extract_operators(qregset, clause) :
    extracted = model.Clause()
    
    for op in clause.ops :
        new_op = None
        if isinstance(op, model.Measure) :
            if op.qreg in qregset :
                new_op = model.Measure(op.qreg, op.outref)
        elif isinstance(op, model.Gate) :
            if op.cntrlist is None :
                # normal gate
                ov = _overlap_1(qregset, op.qreglist)
                if len(ov) != 0 :
                    new_op = op.create(op.qreglist, None)
            else :
                # controlled gate
                overlap_target = _overlap_1(qregset, op.qreglist)
                overlap_control = _overlap_1(qregset, op.cntrlist)
                if len(overlap_target) != 0 or len(overlap_control) != 0 :
                    new_op = op.create(overlap_target, overlap_control)
        elif isinstance(op, model.Clause) :
            new_clause = extract_operators(qregset, op)
            if len(new_clause.ops) != 0 :
                new_op = new_clause
        elif isinstance(op, model.IfClause) :
            new_clause = extract_operators(qregset, op.clause)
            if len(new_clause.ops) != 0 :
                new_op = model.IfClause(op.refs, op.val)
                new_op.set_clause(new_clause)
        elif isinstance(op, (model.Barrier, model.Reset)) :
            qregs = _overlap_1(qregset, op.qregset)
            if len(qregs) != 0 :
                new_op = op.__class__(qregs)
        elif isinstance(op, (FrameBegin, FrameEnd)):
            pass
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
    refset_org = clause.get_refset()
    # update qregset for each clause including nested ones.
    update_clause_registers(clause)
    # get union to get all qregs.
    all_qregset = qregset_org | clause.qregset
    all_refset = refset_org | clause.refset

    # give operators their numbers.
    for idx, op in enumerate(clause.ops) :
        op.set_idx(idx)
        
    if kwargs.get('isolate_circuits', True) :
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
    isolated.set_refset(all_refset)
            
    return isolated
