import qasm.model as qasm
from . import model as sim
import numpy as np
import math
import random


class Simulator :
    def __init__(self, program) :
        self.program = program


    def get_n_circuits(self) :
        return len(self.program.circuits)
        
    def get_qstates(self, idx) :
        return self.qubit_groups[idx]
    
    def get_cregs(self) :
        return self.cregs

    def prepare(self) :
        qubit_groups = []
        ops = []
        for circuit_idx, circuit in enumerate(self.program.circuits) :
            qubit_groups.append(sim.QubitStates(circuit.get_n_qregs()))
            cregs = sim.Cregs(circuit.get_n_cregs())
            ops += [(op, circuit_idx) for op in circuit.ops] 
            
        # FIXME: sort ops
        self.ops = ops
        self.qubit_groups = qubit_groups
        self.cregs = cregs
        
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
        else :
            raise RuntimeError()

    def _measure(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]
        circuit = self.program.circuits[circ_idx]

        for in0, creg in zip(op.in0, op.cregs) :
            lane = circuit.get_creg_lane(creg)
            
            bitmask_lane = 1 << lane
            bitmask_hi = ~((2 << lane) - 1)
            bitmask_lo = (1 << lane) - 1
            n_states = 2 ** (circuit.get_n_qregs() - 1)
            prob = 0.
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                qs = qstates[idx_lo]
                prob += (qs * qs.conj()).real

            if (random.random() < prob) :
                self.cregs[lane] = 0
                norm = 1. / math.sqrt(prob)
                for idx in range(n_states) :
                    idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                    idx_hi = idx_lo | bitmask_lane
                    qstates[idx_lo] *= norm
                    qstates[idx_hi] = 0.
            else :
                self.cregs[lane] = 1
                norm = 1. / math.sqrt(1. - prob)
                for idx in range(n_states) :
                    idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                    idx_hi = idx_lo | bitmask_lane
                    qstates[idx_lo] = 0.
                    qstates[idx_hi] *= norm
                

    def _apply_unary_gate(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]
        circuit = self.program.circuits[circ_idx]

        for in0 in op.in0 :
            lane = circuit.get_qreg_lane(in0)
            bitmask_lane = 1 << lane
            bitmask_hi = ~((2 << lane) - 1)
            bitmask_lo = (1 << lane) - 1
            n_states = 2 ** (circuit.get_n_qregs() - 1)
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qs0 = qstates[idx_lo]
                qs1 = qstates[idx_hi]
                qsout = np.dot(op.get_matrix(), [qs0, qs1])
                qstates[idx_lo] = qsout[0]
                qstates[idx_hi] = qsout[1]

    def _apply_control_gate(self, op, circ_idx) :
        qstates = self.qubit_groups[circ_idx]
        circuit = self.program.circuits[circ_idx]

        for in0, in1 in zip(op.in0, op.in1) :
            lane0 = circuit.get_qreg_lane(in0)
            lane1 = circuit.get_qreg_lane(in1)
            bitmask_control = 1 << lane0
            bitmask_target = 1 << lane1

            bitmask_lane_max = max(bitmask_control, bitmask_target)
            bitmask_lane_min = min(bitmask_control, bitmask_target)
        
            bitmask_hi = ~(bitmask_lane_max * 2 - 1)
            bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1)
            bitmask_lo = bitmask_lane_min - 1
        
            n_states = 1 << (circuit.get_n_qregs() - 2)
            for idx in range(n_states) :
                idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control
                idx_1 = idx_0 | bitmask_target
            
                qs0 = qstates[idx_0]
                qs1 = qstates[idx_1]
            
                qsout = np.dot(op.get_matrix(), [qs0, qs1])
                qstates[idx_0] = qsout[0]
                qstates[idx_1] = qsout[1]


def py(circuit) :
    return Simulator(circuit)
