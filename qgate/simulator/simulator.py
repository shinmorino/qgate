import qasm.model as qasm
from . import model as sim
import numpy as np
import math
import random


def _op_key(op_tuple) :
    return op_tuple[0].idx


class Simulator :
    def __init__(self, program) :
        self.program = program

    def get_n_circuits(self) :
        return len(self.qubit_groups)
        
    def get_qstates(self, idx) :
        return self.qubit_groups[idx]
    
    def get_creg_array_dict(self) :
        return self.creg_array_dict

    def prepare(self) :
        qubit_groups = []
        ops = []

        clauses = self.program.get_circuits()
        
        for circuit_idx, circuit in enumerate(self.program.circuit.clauses) :
            qregs = circuit.get_qregs()
            qubit_groups.append(sim.QubitStates(qregs))
            ops += [(op, circuit_idx) for op in circuit.ops]

        # FIXME: basic operator ordering
        ops = sorted(ops, key = _op_key)
            
            
        creg_array_dict = sim.CregArrayDict(self.program.creg_arrays)
            
        # FIXME: sort ops
        self.ops = ops
        self.qubit_groups = qubit_groups
        self.creg_array_dict = creg_array_dict
        
        self.step_iter = iter(self.ops)

    def run_step(self) :
        try :
            op = next(self.step_iter)
            self._apply_op(op[0], op[1])
            return True
        except StopIteration :
            return False

    def terminate(self) :
        # release resources.
        self.qubit_groups = None
        self.program = None
        self.ops = None
        
    def _apply_op(self, op, circ_idx) :
        if isinstance(op, qasm.Measure) :
            self._measure(op, circ_idx)
        elif isinstance(op, qasm.UnaryGate) :
            self._apply_unary_gate(op, circ_idx)
        elif isinstance(op, qasm.ControlGate) :
            self._apply_control_gate(op, circ_idx)
        elif isinstance(op, qasm.Barrier) :
            pass # Since this simulator runs step-wise, able to ignore barrier.
        elif isinstance(op, qasm.Reset) :
            self._apply_reset(op, circ_idx)
        elif isinstance(op, qasm.Clause) :
            self._apply_clause(op, circ_idx)
        elif isinstance(op, qasm.IfClause) :
            self._apply_if_clause(op, circ_idx)
        else :
            raise RuntimeError()

    def _measure(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]

        for in0, creg in zip(op.in0, op.cregs) :
            lane = qstates.get_lane(in0)
            
            bitmask_lane = 1 << lane
            bitmask_hi = ~((2 << lane) - 1)
            bitmask_lo = (1 << lane) - 1
            n_states = 2 ** (qstates.get_n_lanes() - 1)
            prob = 0.
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                qs = qstates[idx_lo]
                prob += (qs * qs.conj()).real

            if (random.random() < prob) :
                self.creg_array_dict.set(creg, 0)
                norm = 1. / math.sqrt(prob)
                for idx in range(n_states) :
                    idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                    idx_hi = idx_lo | bitmask_lane
                    qstates[idx_lo] *= norm
                    qstates[idx_hi] = 0.
            else :
                self.creg_array_dict.set(creg, 1)
                norm = 1. / math.sqrt(1. - prob)
                for idx in range(n_states) :
                    idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                    idx_hi = idx_lo | bitmask_lane
                    qstates[idx_lo] = 0.
                    qstates[idx_hi] *= norm

    def _apply_reset(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]

        for qreg in op.qregset :
            lane = qstates.get_lane(qreg)
            bitmask_lane = 1 << lane
            bitmask_hi = ~((2 << lane) - 1)
            bitmask_lo = (1 << lane) - 1
            n_states = 2 ** (qstates.get_n_lanes() - 1)

            prob = 0.
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                qs_lo = qstates[idx_lo]
                prob += (qs_lo * qs_lo.conj()).real

            # Assuming reset is able to be applyed after measurement.
            # Ref: https://quantumcomputing.stackexchange.com/questions/3908/possibility-of-a-reset-quantum-gate
            # FIXME: add a mark to qubit that tells if it entangles or not.
            if prob == 0. :
                # prob == 0 means previous measurement gave creg = 1.
                # negating this qubit
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] = qstates[idx_hi]
                qstates[idx_hi] = 0.
            else :
                assert False, "Is traceout suitable?"
                norm = math.sqrt(1. / prob)
                for idx in range(n_states) :
                    idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                    idx_hi = idx_lo | bitmask_lane
                    qstates[idx_lo] *= norm
                    qstates[idx_hi] = 0.

    def _apply_if_clause(self, op, circ_idx) :
        if self.creg_array_dict.get_as_bits(op.creg_array) == op.val :
            self._apply_op(op.clause, circ_idx)

    def _apply_clause(self, op, circ_idx) :
        for clause_op in op.ops :
            self._apply_op(clause_op, circ_idx)

    def _apply_unary_gate(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]

        for in0 in op.in0 :
            lane = qstates.get_lane(in0)
            bitmask_lane = 1 << lane
            bitmask_hi = ~((2 << lane) - 1)
            bitmask_lo = (1 << lane) - 1
            n_states = 2 ** (qstates.get_n_lanes() - 1)
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qs0 = qstates[idx_lo]
                qs1 = qstates[idx_hi]
                qsout = np.matmul(op.get_matrix(), np.matrix([qs0, qs1], np.complex128).T)
                qstates[idx_lo] = qsout[0]
                qstates[idx_hi] = qsout[1]

    def _apply_control_gate(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]

        for in0, in1 in zip(op.in0, op.in1) :
            lane0 = qstates.get_lane(in0)
            lane1 = qstates.get_lane(in1)
            bitmask_control = 1 << lane0
            bitmask_target = 1 << lane1

            bitmask_lane_max = max(bitmask_control, bitmask_target)
            bitmask_lane_min = min(bitmask_control, bitmask_target)
        
            bitmask_hi = ~(bitmask_lane_max * 2 - 1)
            bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1)
            bitmask_lo = bitmask_lane_min - 1
        
            n_states = 1 << (qstates.get_n_lanes() - 2)
            for idx in range(n_states) :
                idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control
                idx_1 = idx_0 | bitmask_target
            
                qs0 = qstates[idx_0]
                qs1 = qstates[idx_1]
            
                qsout = np.matmul(op.get_matrix(), np.matrix([qs0, qs1], np.complex128).T)
                qstates[idx_0] = qsout[0]
                qstates[idx_1] = qsout[1]


def py(circuit) :
    return Simulator(circuit)
