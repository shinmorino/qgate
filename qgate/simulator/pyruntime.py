import numpy as np
import math


def _abs2(c) :
    return c.real ** 2 + c.imag ** 2

def _null(c) :
    return c

# representing a single qubit or entangled qubits.
class QubitStates :
    def __init__(self, processor) :
        self._qproc = processor

    def deallocate(self) :
        self.states = None

    def get_n_qregs(self) :
        return len(self.qreglist)

    # internal methods
    
    def __getitem__(self, key) :
        return self.states[key]

    def __setitem__(self, key, value) :
        self.states[key] = value

    def get_lane(self, qreg) :
        return self.qreglist.index(qreg)
    
    def convert_to_local_lane_idx(self, idx) :
        local_idx = 0
        for bitpos, qreg in enumerate(self.qreglist) :
            if ((1 << qreg.id) & idx) != 0 :
                local_idx |= 1 << bitpos
        return local_idx
    
    def get_state_by_global_idx(self, idx) :
        local_idx = self.convert_to_local_lane_idx(idx)
        return self[local_idx]


class PyQubitProcessor :

    def clear(self) :
        pass

    def prepare(self) :
        pass
    
    def initialize_qubit_states(self, qregs, qstates, n_lanes_per_chunk, device_ids) :
        qstates.qreglist = list(qregs)
        qstates.states = np.empty([2 ** len(qregs)], np.complex128)
        
    def reset_qubit_states(self, qstates) :
        qstates.states[:] = np.complex128()
        qstates.states[0] = 1
        
    def measure(self, rand_num, qstates, qreg) :

        lane = qstates.get_lane(qreg)

        bitmask_lane = 1 << lane
        bitmask_hi = ~((2 << lane) - 1)
        bitmask_lo = (1 << lane) - 1
        n_states = 2 ** (qstates.get_n_qregs() - 1)
        prob = 0.
        for idx in range(n_states) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            qs = qstates[idx_lo]
            prob += (qs * qs.conj()).real

        if (rand_num < prob) :
            creg_value = 0
            norm = 1. / math.sqrt(prob)
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] *= norm
                qstates[idx_hi] = 0.
        else :
            creg_value = 1
            norm = 1. / math.sqrt(1. - prob)
            for idx in range(n_states) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] = 0.
                qstates[idx_hi] *= norm

        return creg_value

    def apply_reset(self, qstates, qreg) :

        lane = qstates.get_lane(qreg)
        bitmask_lane = 1 << lane
        bitmask_hi = ~((2 << lane) - 1)
        bitmask_lo = (1 << lane) - 1
        n_states = 2 ** (qstates.get_n_qregs() - 1)

        for idx in range(n_states) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            idx_hi = idx_lo | bitmask_lane
            qstates[idx_lo] = qstates[idx_hi]
            qstates[idx_hi] = 0.


    def apply_unary_gate(self, mat, qstates, qreg) :
        lane = qstates.get_lane(qreg)
        bitmask_lane = 1 << lane
        bitmask_hi = ~((2 << lane) - 1)
        bitmask_lo = (1 << lane) - 1
        n_states = 2 ** (qstates.get_n_qregs() - 1)
        for idx in range(n_states) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            idx_hi = idx_lo | bitmask_lane
            qs0 = qstates[idx_lo]
            qs1 = qstates[idx_hi]
            qsout = np.matmul(mat, np.array([qs0, qs1], np.complex128).T)
            qstates[idx_lo] = qsout[0]
            qstates[idx_hi] = qsout[1]


    def apply_control_gate(self, mat, qstates, control, target) :
        lane0 = qstates.get_lane(control)
        lane1 = qstates.get_lane(target)
        bitmask_control = 1 << lane0
        bitmask_target = 1 << lane1

        bitmask_lane_max = max(bitmask_control, bitmask_target)
        bitmask_lane_min = min(bitmask_control, bitmask_target)

        bitmask_hi = ~(bitmask_lane_max * 2 - 1)
        bitmask_mid = (bitmask_lane_max - 1) & ~((bitmask_lane_min << 1) - 1)
        bitmask_lo = bitmask_lane_min - 1

        n_states = 1 << (qstates.get_n_qregs() - 2)
        for idx in range(n_states) :
            idx_0 = ((idx << 2) & bitmask_hi) | ((idx << 1) & bitmask_mid) | (idx & bitmask_lo) | bitmask_control
            idx_1 = idx_0 | bitmask_target

            qs0 = qstates[idx_0]
            qs1 = qstates[idx_1]
            qsout = np.matmul(mat, np.array([qs0, qs1], np.complex128).T)
            qstates[idx_0] = qsout[0]
            qstates[idx_1] = qsout[1]

    def get_states(self, values, mathop, qubit_states_list) :
        for idx in range(values.shape[0]) :
            val = 1.
            for qstates in qubit_states_list :
                state = qstates.get_state_by_global_idx(idx)
                val *= mathop(state)
            values[idx] = val

def create_qubit_states(dtype, processor) :
    return QubitStates(processor)

def create_qubit_processor(dtype) :
    return PyQubitProcessor()
