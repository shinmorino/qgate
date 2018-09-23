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


def seperate_programs(program) :
    seperated = model.Program()
    seperated.set_regs(program.qregs.copy(), program.creg_arrays.copy(), program.cregs.copy())

    single_qregs = program.qregs.copy()
    circuits = seperate_circuit(program.circuit)
    single_qregs -= program.circuit.qregs

    for qreg in single_qregs :
        circuit = model.Clause()
        circuit.qregs = set([qreg])
        circuits.append(circuit)

    seperated.circuit = circuits
    return seperated


def seperate_circuit(circuit) :

    circuits = model.IsolatedClauses()
    
    pairs = []
    nodesets = []

    # extract qubit groups
    
    # extract pairs
    for op in circuit.ops :
        if isinstance(op, model.ControlGate) :
            for in0, in1 in zip(op.in0, op.in1) :
                pairs.append(set([in0, in1]))
                
    # marge pairs
    while len(pairs) != 0 :
        nodeset = pairs[0]
        unused_pairs = []
        del pairs[0]
        for counter_pair in pairs :
            if len(nodeset & counter_pair) != 0 :
                # has overlap
                nodeset = nodeset | counter_pair
            else :
                unused_pairs.append(counter_pair)
        nodesets.append(nodeset)
        pairs = unused_pairs

    # used qubits
    used_qregs = set()
    for nodeset in nodesets :
        used_qregs |= nodeset

    # collect remaining qregs
    remaining_qregs = circuit.qregs - used_qregs
    for qreg in remaining_qregs :
        nodesets.append(set([qreg]))
            
    # extract ops according to node groups
    ops = circuit.ops
    for nodeset in nodesets :
        new_circuit = model.Clause()
        unused_ops = []
        cregs = set()
        for op in ops :
            used = False
            if isinstance(op, model.Measure) :
                if _overlapped(op.in0, nodeset) :
                    new_circuit.add_op(op)
                    used = True
            elif isinstance(op, model.UnaryGate) :
                if _overlapped(op.in0, nodeset) :
                    new_circuit.add_op(op)
                    used = True
            elif isinstance(op, model.ControlGate) :
                if _overlapped(op.in0, nodeset) and _overlapped(op.in1, nodeset) :
                    new_circuit.add_op(op)
                    used = True
                else :
                    assert not _overlapped(op.in0, nodeset) and not _overlapped(op.in1, nodeset)
            elif isinstance(op, model.Clause) :
                if _overlapped(op.qregs, nodeset) :
                    new_circuit.add_op(op)
            elif isinstance(op, model.IfClause) :
                if _overlapped(op.clause.qregs, nodeset) :
                    new_circuit.add_op(op)
            elif isinstance(op, (model.Barrier, model.Reset)) : 
                if _overlapped(op.qregset, nodeset) :
                    new_circuit.add_op(op)
            else :
                raise RuntimeError()

            if not used :
                unused_ops.append(op)

        # In case this qreg does not have any gate.
        new_circuit.qregs |= nodeset
        circuits.append(new_circuit)

        ops = unused_ops
        
    return circuits


def update_register_references(circuit) :
    qregs, cregs = set(), set()
    for op in circuit.ops :
        if isinstance(op, model.Measure) :
            qregs |= set(op.in0)
            cregs |= set(op.cregs)
        elif isinstance(op, model.UnaryGate) :
            qregs |= set(op.in0)
        elif isinstance(op, model.ControlGate) :
            qregs |= set(op.in0 + op.in1)
        elif isinstance(op, (model.Barrier, model.Reset)) :
            qregs |= op.qregset
        elif isinstance(op, model.Clause) :
            update_register_references(op)
            qregs_clause, cregs_clause = op.get_regs()
            qregs |= qregs_clause
            cregs |= cregs_clause
        elif isinstance(op, model.IfClause) :
            update_register_references(op.clause)
            qregs_clause, cregs_clause = op.clause.get_regs()
            qregs |= qregs_clause
            cregs |= cregs_clause
        else :
            raise RuntimeError()
        
    circuit.set_regs(qregs, cregs)

def process(program, **kwargs) :
    update_register_references(program.circuit)
    
    if 'seperate_circuit' in kwargs.keys() :
        if kwargs['seperate_circuit'] :
            program = seperate_programs(program)
    if 'expand_register_list' in kwargs.keys() :
        if kwargs['expand_register_list'] :
            program = expand_register_list(program)
            
    return program
