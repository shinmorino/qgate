import qasm.model as qasm
from .pykernel import PyKernel
import numpy as np
import math
import random
            

def _op_key(op_tuple) :
    return op_tuple[0].idx


# creg arrays and their values.
class CregDict :
    def __init__(self, creg_arrays) :
        self.creg_dict = dict()
        for creg_array in creg_arrays :
            values = np.zeros([creg_array.length()])
            self.creg_dict[creg_array] = values

    def get_arrays(self) :
        return self.creg_dict.keys()
    
    def set_value(self, creg, value) :
        values = self.creg_dict[creg.creg_array]
        values[creg.idx] = value
        
    def get_value(self, creg) :
        values = self.creg_dict[creg.creg_array]
        return values[creg.idx]

    def get_values(self, creg_array) :
        return self.creg_dict[creg_array]

    def get_array_as_integer(self, creg_array) :
        values = self.creg_dict[creg_array]
        ivalue = 0
        for idx, value in enumerate(values) :
            if value == 1 :
                ivalue |= 1 << idx
        return ivalue


class Simulator :
    def __init__(self, program) :
        self.program = program
        self.kernel = PyKernel(program.qregs)

    def get_qubits(self) :
        return self.kernel.get_qubits()
    
    def get_creg_dict(self) :
        return self.creg_dict

    def prepare(self) :
        ops = []

        clauses = self.program.get_circuits()
        for circuit_idx, circuit in enumerate(clauses) :
            ops += [(op, circuit_idx) for op in circuit.ops]

        # FIXME: basic operator ordering
        ops = sorted(ops, key = _op_key)
        self.ops = ops
        
        for circuit_idx, circuit in enumerate(clauses) :
            self.kernel.set_circuit(circuit_idx, circuit)
        
        self.creg_dict = CregDict(self.program.creg_arrays)
        
        self.step_iter = iter(self.ops)

    def run_step(self) :
        try :
            op = next(self.step_iter)
            self._apply_op(op[0], op[1])
            return True
        except StopIteration :
            return False

    def run(self) :
        while self.run_step() :
            pass

    def terminate(self) :
        # release resources.
        self.qubit_groups = None
        self.creg_array_dict = None
        self.program = None
        self.ops = None
        
    def _apply_op(self, op, circ_idx) :
        if isinstance(op, qasm.Clause) :
            self._apply_clause(op, circ_idx)
        elif isinstance(op, qasm.IfClause) :
            self._apply_if_clause(op, circ_idx)
        elif isinstance(op, qasm.Measure) :
            self._measure(op, circ_idx)
        elif isinstance(op, qasm.UnaryGate) :
            self._apply_unary_gate(op, circ_idx)
        elif isinstance(op, qasm.ControlGate) :
            self._apply_control_gate(op, circ_idx)
        elif isinstance(op, qasm.Barrier) :
            pass  # Since this simulator runs step-wise, able to ignore barrier.
        elif isinstance(op, qasm.Reset) :
            self._apply_reset(op, circ_idx)
        else :
            assert False, "Unknown operator."

    def _apply_if_clause(self, op, circ_idx) :
        if self.creg_dict.get_array_as_integer(op.creg_array) == op.val :
            self._apply_op(op.clause, circ_idx)

    def _apply_clause(self, op, circ_idx) :
        for clause_op in op.ops :
            self._apply_op(clause_op, circ_idx)
    
    def _measure(self, op, circ_idx) :
        for in0, creg in zip(op.in0, op.cregs) :
            rand_num = random.random()
            creg_value = self.kernel.measure(rand_num, circ_idx, in0)
            self.creg_dict.set_value(creg, creg_value)

    def _apply_reset(self, op, circ_idx) :
        for qreg in op.qregset :
            self.kernel.apply_reset(circ_idx, qreg)
                    
    def _apply_unary_gate(self, op, circ_idx) :
        for in0 in op.in0 :
            self.kernel.apply_unary_gate(op.get_matrix(), circ_idx, in0)

    def _apply_control_gate(self, op, circ_idx) :
        for in0, in1 in zip(op.in0, op.in1) :
            self.kernel.apply_control_gate(op.get_matrix(), circ_idx, in0, in1)


def py(circuit) :
    return Simulator(circuit)
