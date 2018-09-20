import qasm_model as qasm


def _overlapped(arr0, arr1) :
    return len(set(arr0) & set(arr1)) != 0

        
def expand_register_lists(program) :
    expanded = qasm.Program()
    circuits = program.circuits
    
    expanded.qregs = program.qregs.copy()
    expanded.cregs = program.cregs.copy()
    
    for circuit in circuits :
        new_circuit = qasm.Circuit()
        for op in circuit.ops :
            if isinstance(op, qasm.Measure) :
                for qreg, creg in zip(op.in0, op.cregs) :
                    cloned = type(op)([qreg], [creg])
                    new_circuit.add_op(cloned)
            elif isinstance(op, qasm.UnaryGate) :
                for qreg in op.in0 :
                    cloned = type(op)([qreg])
                    new_circuit.add_op(cloned)
            elif isinstance(op, qasm.ControlGate) :
                for qreg0, qreg1 in zip(op.in0, op.in1) :
                    cloned = type(op)([qreg0], [qreg1])
                    new_circuit.add_op(cloned)
            else :
                raise RuntimeError()
            
        expanded.add_circuit(new_circuit)
    
    return expanded


def seperate_programs(program) :
    seperated = qasm.Program()
    seperated.set_regs(program.qregs, program.cregs)

    circuits = []
    single_qregs = program.qregs.copy()
    for circuit in program.circuits :
        single_qregs -= circuit.qregs
        circuits += seperate_circuit(circuit)

    for qreg in single_qregs :
        circuit = qasm.Circuit()
        circuit.qregs = set([qreg])
        circuits.append(circuit)

    seperated.circuits = circuits
    return seperated


def seperate_circuit(circuit) :

    circuits = []
    
    pairs = []
    nodesets = []

    # extract qubit groups
    
    # extract pairs
    for op in circuit.ops :
        if isinstance(op, qasm.ControlGate) :
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
        new_circuit = qasm.Circuit()
        unused_ops = []
        cregs = set()
        for op in ops :
            used = False
            if isinstance(op, qasm.Measure) :
                if _overlapped(op.in0, nodeset) :
                    new_circuit.add_op(op)
                    used = True
            elif isinstance(op, qasm.UnaryGate) :
                if _overlapped(op.in0, nodeset) :
                    new_circuit.add_op(op)
                    used = True
            elif isinstance(op, qasm.ControlGate) :
                if _overlapped(op.in0, nodeset) and _overlapped(op.in1, nodeset) :
                    new_circuit.add_op(op)
                    used = True
                else :
                    assert not _overlapped(op.in0, nodeset) and not _overlapped(op.in1, nodeset)
            else :
                raise RuntimeError()

            if not used :
                unused_ops.append(op)

        # In case this qreg does not have any gate.
        new_circuit.qregs |= nodeset
        circuits.append(new_circuit)

        ops = unused_ops
        
    return circuits


def _schedule_simple(circuit) :
    circuit.ops = []
    for opflow in circuit.program.opflows :
        circuit.ops += opflow.ops
