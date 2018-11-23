from . import model


def _overlapped(arr0, arr1) :
    return len(set(arr0) & set(arr1)) != 0

def merge_qreg_groups(groups) :

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


def isolate_circuits(program) :
    seperated = model.Program()
    seperated.set_regs(program.qregs.copy(), program.creg_arrays.copy(), program.cregs.copy())

    single_qregs = program.qregs.copy()
    isolated_clauses = isolate_clauses(program.clause)
    single_qregs -= program.clause.qregs

    for qreg in single_qregs :
        clause = model.Clause()
        clause.qregs = set([qreg])
        isolated_clauses.append(clause)

    seperated.clause = isolated_clauses
    return seperated


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

    unused_qregs = clause.qregs - used_qregs
    for qreg in unused_qregs :
        qreg_groups.append(set([qreg]))

    qreg_groups = merge_qreg_groups(qreg_groups)
        
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
    
    
def _correspondings(qregset, qregs, regs) :
    extracted = []
    for qreg, reg in zip(qregs, regs) :
        if qreg in qregset :
            extracted.append(reg)
    return extracted


def extract_operators(qregset, clause) :

    extracted = model.Clause()
    
    for op in clause.ops :
        new_op = None
        if isinstance(op, model.Measure) :
            in0 = _overlap_1(qregset, op.in0)
            cregs = _correspondings(in0, op.in0, op.cregs)
            if len(in0) != 0 :
                new_op = model.Measure(in0, cregs)
        elif isinstance(op, model.U) :
            in0 = _overlap_1(qregset, op.in0)
            if len(in0) != 0 :
                new_op = op.__class__(op._theta, op._phi, op._lambda, in0)
        elif isinstance(op, model.UnaryGate) :
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
        
    extracted.set_qregs(qregset)
    return extracted


def isolate_clauses(clause) :

    isolated_clauses = model.IsolatedClauses()

    qreg_groups = isolate_qreg_groups(clause)
    # used qubits
    used_qregs = set()
    for links in qreg_groups :
        used_qregs |= links
    # collect unused qregs
    unused_qregs = clause.qregs - used_qregs

    for links in qreg_groups :
        extracted = extract_operators(links, clause)
        isolated_clauses.append(extracted)
    
    for qreg in unused_qregs :
        clause_1_qubit = model.Clause()
        clause_1_qubit.set_qregs(set([qreg]))
        isolated_clauses.append(clause_1_qubit)
        
    return isolated_clauses


def update_register_references(clause) :
    qregs, cregs = set(), set()
    for op in clause.ops :
        if isinstance(op, model.Measure) :
            qregs |= set(op.in0)
        elif isinstance(op, model.UnaryGate) :
            qregs |= set(op.in0)
        elif isinstance(op, model.ControlGate) :
            qregs |= set(op.in0 + op.in1)
        elif isinstance(op, (model.Barrier, model.Reset)) :
            qregs |= op.qregset
        elif isinstance(op, model.Clause) :
            update_register_references(op)
            qregs_clause = op.get_qregs()
            qregs |= qregs_clause
        elif isinstance(op, model.IfClause) :
            update_register_references(op.clause)
            qregs_clause = op.clause.get_qregs()
            qregs |= qregs_clause
        else :
            raise RuntimeError()
        
    clause.set_qregs(qregs)

def process(program, **kwargs) :
    update_register_references(program.clause)

    for idx, op in enumerate(program.clause.ops) :
        op.set_idx(idx)
    
    if 'isolate_circuits' in kwargs.keys() :
        if kwargs['isolate_circuits'] :
            program = isolate_circuits(program)
    if 'expand_register_list' in kwargs.keys() :
        if kwargs['expand_register_list'] :
            program = expand_register_list(program)
            
    return program
