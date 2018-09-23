from . import model


def _overlapped(arr0, arr1) :
    return len(set(arr0) & set(arr1)) != 0

        
def expand_register_lists(program) :
    expanded = model.Program()
    circuits = program.circuits
    
    expanded.qregs = program.qregs.copy()
    expanded.cregs = program.cregs.copy()
    
    for circuit in circuits :
        new_circuit = model.Circuit()
        for op in circuit.ops :
            if isinstance(op, model.Measure) :
                for qreg, creg in zip(op.in0, op.cregs) :
                    cloned = type(op)([qreg], [creg])
                    new_circuit.add_op(cloned)
            elif isinstance(op, model.UnaryGate) :
                for qreg in op.in0 :
                    cloned = type(op)([qreg])
                    new_circuit.add_op(cloned)
            elif isinstance(op, model.ControlGate) :
                for qreg0, qreg1 in zip(op.in0, op.in1) :
                    cloned = type(op)([qreg0], [qreg1])
                    new_circuit.add_op(cloned)
            else :
                raise RuntimeError()
            
        expanded.add_circuit(new_circuit)
    
    return expanded

def merge_qreg_groups(groups) :

    merged = []
    
    while len(groups) != 0 :
        unmerged = []
        
        group = groups0[0]
        groups0.remove(0)

        for other in groups :
            intersect = group0 & other
            if len(intersect) == 0 :
                unmerged.append(other)

        merged.append(group)
        groups = unmerged

    return merged


def isolate_circuits(program) :
    seperated = model.Program()
    seperated.set_regs(program.qregs.copy(), program.creg_arrays.copy(), program.cregs.copy())

    single_qregs = program.qregs.copy()
    circuits = isolate_clauses(program.circuit)
    single_qregs -= program.circuit.qregs

    for qreg in single_qregs :
        circuit = model.Clause()
        circuit.qregs = set([qreg])
        circuits.append(circuit)

    seperated.circuit = circuits
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
    
    
def _corresponds(qregset, qregs, regs) :
    extracted = []
    for qreg, reg in zip(qregs, regs) :
        if qreg in qregset :
            extracted.append(reg)
    return extracted


def extract_operators(clause, qregset) :

    extracted = model.Clause()
    
    for op in clause.ops :
        if isinstance(op, model.Measure) :
            in0 = _overlap_1(qregs, op.in0)
            cregs = _correspondings(in0, op.in0, op.cregs)
            if len(in0) != 0 :
                new_op = model.Measure(in0, cregs)
                extracted.add_op(new_op)
        elif isinstance(op, model.UnaryGate) :
            in0 = _overlap_1(qregset, op.in0)
            if len(in0) != 0 :
                new_op = type(op)(in0)
                extracted.add_op(new_op)
        elif isinstance(op, model.ControlGate) :
            control, target = _overlap_2(qregset, op.in0, op.in1)
            if len(control) != 0 :
                new_op = type(op)(control, target)
                extracted.add_op(new_op)
        elif isinstance(op, model.Clause) :
            new_clause= extract_operations(op)
            if len(new_clause.ops) != 0 :
                extracted.add_op(new_clause)
        elif isinstance(op, model.IfClause) :
            new_clause = extract_operations(op.clause)
            if len(new_clause.ops) != 0 :
                new_if_clause = model.IfClause(op.creg_array, op.val)
                new_if_clause.set_clause(new_clause)
                extracted.add_op(new_if_clause)
        elif isinstance(op, (model.Barrier, model.Reset)) : 
            if _overlapped(op.qregset, nodeset) :
                extracted.add_op(op)
        else :
            raise RuntimeError("Unknown operator")

    extracted.set_qregs(qregset)
    return extracted


def isolate_clauses(circuit) :

    circuits = model.IsolatedClauses()

    qreg_groups = isolate_qreg_groups(circuit)
    # used qubits
    used_qregs = set()
    for links in qreg_groups :
        used_qregs |= links
    # collect unused qregs
    unused_qregs = circuit.qregs - used_qregs

    for links in qreg_groups :
        clause = extract_operators(circuit, links)
        circuits.append(clause)
    
    for qreg in unused_qregs :
        clause = model.Clause()
        clause.set_qregs(set([qreg]))
        circuits.append(clause)
        
    return circuits


def update_register_references(circuit) :
    qregs, cregs = set(), set()
    for op in circuit.ops :
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
        
    circuit.set_qregs(qregs)

def process(program, **kwargs) :
    update_register_references(program.circuit)
    
    if 'seperate_circuit' in kwargs.keys() :
        if kwargs['seperate_circuit'] :
            program = isolate_circuits(program)
    if 'expand_register_list' in kwargs.keys() :
        if kwargs['expand_register_list'] :
            program = expand_register_list(program)
            
    return program
