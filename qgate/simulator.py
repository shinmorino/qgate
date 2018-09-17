import qasm_model as qasm
import sim_model as sim
import numpy as np

class Simulator :
    def __init__(self, circuit) :
        self.circuit = circuit

    def get_qstates(self, idx) :
        return self.qubit_groups[idx]

    def prepare(self) :
        qubit_groups = []
        ops = []
        for programIdx, program in enumerate(self.circuit.programs) :
            qubit_groups.append(sim.QubitStates(program.get_n_qregs()))
            ops += [(op, programIdx) for op in program.ops] 
            
        # FIXME: sort ops
        self.ops = ops
        self.qubit_groups = qubit_groups

        self.step_iter = iter(self.ops)

    def run_step(self) :
        try :
            op = next(self.step_iter)
            self._apply_op(op[0], op[1])
            return True
        except StopIteration :
            return False

    def terminate(self) :
        pass
        
    def _apply_op(self, op, programIdx) :
        if isinstance(op, qasm.Measure) :
            self._measure(op, programIdx)
        elif isinstance(op, qasm.UnaryGate) :
            self._apply_unary_gate(op, programIdx)
        elif isinstance(op, qasm.ControlGate) :
            self._apply_control_gate(op, programIdx)
        else :
            raise RuntimeError()

    def _measure(self, op, circuitIdx) :
        pass

    def _apply_unary_gate(self, op, programIdx) :
        qstates = self.qubit_groups[programIdx]
        
        lane = self.circuit.get_qreg_lane(op.in0)
        bitmask_lane = 1 << lane
        bitmask_hi = ~((2 << lane) - 1)
        bitmask_lo = (1 << lane) - 1
        n_states = 2 ** (self.circuit.get_n_qregs() - 1)
        for idx in range(n_states) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            idx_hi = idx_lo | bitmask_lane
            qs0 = qstates[idx_lo]
            qs1 = qstates[idx_hi]
            qsout = np.dot(op.get_matrix(), [qs0, qs1])
            qstates[idx_lo] = qsout[0]
            qstates[idx_hi] = qsout[1]

    def _apply_control_gate(self, op, programIdx) :
        qstates = self.qubit_groups[programIdx]
        
        lane0 = self.circuit.get_qreg_lane(op.in0)
        lane1 = self.circuit.get_qreg_lane(op.in1)
        bitmask_control = 1 << lane0
        bitmask_target = 1 << lane1

        bitmask_lane_max = max(bitmask_control, bitmask_target)
        bitmask_lane_min = min(bitmask_control, bitmask_target)
        
        bitmask_hi = ~(bitmask_lane_max * 2 - 1)
        bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1)
        bitmask_lo = bitmask_lane_min - 1
        
        n_states = 1 << (self.circuit.get_n_qregs() - 2)
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
