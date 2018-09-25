import qasm.model as qasm
from .pykernel import PyKernel
import numpy as np
import math
import random
            

def _op_key(op_tuple) :
    return op_tuple[0].idx


class Simulator :
    def __init__(self, program) :
        self.program = program
        self.kernel = PyKernel(program.qregs)

    def get_kernel_state_accessor(self) :
        return self.kernel.get_state_accessor()

    def get_probability_list(self) :
        acc = self.kernel.get_state_accessor()
        return acc.get_probability_list()
    
    def get_creg_dict(self) :
        acc = self.kernel.get_state_accessor()
        return acc.get_creg_dict()

    def prepare(self) :
        ops = []

        clauses = self.program.get_circuits()
        for circuit_idx, circuit in enumerate(clauses) :
            ops += [(op, circuit_idx) for op in circuit.ops]

        # FIXME: basic operator ordering
        ops = sorted(ops, key = _op_key)
        self.ops = ops
        
        for circuit_idx, circuit in enumerate(clauses) :
            self.kernel.set_qregset(circuit_idx, circuit.get_qregs())
        self.kernel.set_creg_arrays(self.program.creg_arrays)
        
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
        else :
            self.kernel.apply_op(op, circ_idx)

    def _apply_if_clause(self, op, circ_idx) :
        if self.kernel.get_creg_array_as_bits(op.creg_array) == op.val :
            self._apply_op(op.clause, circ_idx)

    def _apply_clause(self, op, circ_idx) :
        for clause_op in op.ops :
            self._apply_op(clause_op, circ_idx)

def py(circuit) :
    return Simulator(circuit)
