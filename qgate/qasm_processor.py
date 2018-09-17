import qasm_model as qasm
import circuit as circ

def expand_register_lists(program) :
    expanded = qasm.Program()
    ops = program.ops
    for op in ops :
        if isinstance(op, qasm.Measure) :
            for qreg, creg in zip(op.in0, op.cregs) :
                cloned = type(op)(qreg, creg)
                expanded.ops.append(cloned)
        elif isinstance(op, qasm.UnaryGate) :
            for qreg in op.in0 :
                cloned = type(op)(qreg)
                expanded.ops.append(cloned)
        elif isinstance(op, qasm.ControlGate) :
            for qreg0, qreg1 in zip(op.in0, op.in1) :
                cloned = type(op)(qreg0, qreg1)
                expanded.ops.append(cloned)
        else :
            raise RuntimeError()
    
    for idx, op in enumerate(expanded.ops) :
        op.idx = idx
    
    expanded.qregs = program.qregs
    expanded.cregs = program.cregs
    return expanded


def seperate_programs(program) :
    programs = []
    pairs = []
    nodesets = []

    for op in program.ops :
        if isinstance(op, qasm.ControlGate) :
            pairs.append(set([op.in0, op.in1]))
    
    # extract node groups
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

    # used nodes
    used_qregs = set()
    for nodeset in nodesets :
        used_qregs |= nodeset
        
    # add single nodes
    for qreg in program.qregs :
        if not qreg in used_qregs :
            nodesets.append(set([qreg]))

    # extract ops according to node groups
    ops = program.ops
    for nodeset in nodesets :
        used_ops, unused_ops = [], []
        cregs = set()
        for op in program.ops :
            if isinstance(op, qasm.Measure) :
                if op.in0 in nodeset :
                    used_ops.append(op)
                    cregs |= set([op.cregs])
                else :
                    unused_ops.append(op)
            elif isinstance(op, qasm.UnaryGate) :
                if op.in0 in nodeset :
                    used_ops.append(op)
                else :
                    unused_ops.append(op)
            elif isinstance(op, qasm.ControlGate) :
                if (op.in0 in nodeset) and (op.in1 in nodeset) :
                    used_ops.append(op)
                else :
                    assert not (op.in0 in nodeset) and not (op.in1 in nodeset)
                    unused_ops.append(op)
            else :
                raise RuntimeError()
                        
        program = qasm.Program()
        program.qregs = list(nodeset)
        program.cregs = list(cregs)
        program.ops = used_ops

        programs.append(program)
        ops = unused_ops

                
    return programs


def map_to_circuit(programs) :
    circuit = circ.Circuit()
    circuit.programs = programs
    for program in programs :
        # map registers at first
        for qreg in program.qregs :
            circuit.map_qreg(qreg)
        for creg in program.cregs :
            circuit.map_creg(creg)

    _schedule_simple(circuit)
    return circuit


def _schedule_simple(circuit) :
    circuit.ops = []
    for program in circuit.programs :
        circuit.ops += program.ops
