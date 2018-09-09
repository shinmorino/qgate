import qasm_model as qasm
import circuit as circ

def expand_register_lists(program) :
    expanded = qasm.Program()
    ops = program.ops
    for op in ops :
        if isinstance(op, qasm.Measure) :
            for qreg, creg in zip(op.qregs, op.cregs) :
                cloned = type(op)(qreg, creg)
                expanded.ops.append(cloned)
        elif op.get_n_inputs() == 1 :
            for qreg in op.in0 :
                cloned = type(op)(qreg)
                expanded.ops.append(cloned)
        else :
            for qreg0, qreg1 in zip(op.in0, op.in1) :
                cloned = type(op)(qreg0, qreg1)
                expanded.ops.append(cloned)
    
    for idx, op in enumerate(expanded.ops) :
        op.idx = idx
    
    return expanded

        
def seperate_programs(program) :
    programs = []
    pairs = []
    nodesets = []
    
    for op in program.ops :
        if op.get_n_inputs() == 2 :
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

    # extract ops according to node groups
    ops = program.ops
    for nodeset in nodesets :
        used_ops, unused_ops = [], []
        qregs, cregs = set(), set()
        for op in ops :
            if isinstance(op, qasm.Measure) :
                if op.qregs in nodeset :
                    used_ops.append(op)
                    cregs.add(op.cregs)
                else :
                    unused_ops.append(op)
            elif op.get_n_inputs() == 1 :
                if op.in0 in nodeset :
                    used_ops.append(op)
                    qregs.add(op.in0)
                else :
                    unused_ops.append(op)
            else : # elif gate.get_n_inputs() == 2 :
                if (op.in0 in nodeset) and (op.in1 in nodeset) :
                    used_ops.append(op)
                    qregs.add(op.in0)
                    qregs.add(op.in1)
                else :
                    assert not (op.in0 in nodeset) and not (op.in1 in nodeset)
                    unused_ops.append(op)
                        
        program = qasm.Program()
        program.qregs = list(qregs)
        program.cregs = list(cregs)
        program.ops = used_ops

        programs.append(program)
        ops = unused_ops

                
    return programs


def map_to_circuits(programs) :
    circuits = []
    for program in programs :
        circuit = circ.Circuit()
        _map_program_simple(circuit, program)
        circuits.append(circuit)

    return circuits


def _map_program_simple(circuit, program) :
    ops = []

    for op in program.ops :
        if isinstance(op, qasm.Measure) :
            op.qreg_idx = circuit.map_qreg(op.qregs)
            op.creg_idx = circuit.map_creg(op.cregs)
            ops.append(op)
        elif op.get_n_inputs() == 1 :
            op.in0_idx = circuit.map_qreg(op.in0)
            ops.append(op)
        else :
            assert op.get_n_inputs() == 2
            op.in0_idx = circuit.map_qreg(op.in0)
            op.in1_idx = circuit.map_qreg(op.in1)
            ops.append(op)

    circuit.ops = ops

