import numpy as np
import math

def adjoint(mat) :
    return np.conjugate(mat.T)

def _abs2(c) :
    return c.real ** 2 + c.imag ** 2

def _null(c) :
    return c

# representing a single qubit or entangled qubits.
class QubitStates :
    
    def allocate(self, n_lanes) :
        self.n_lanes = n_lanes
        self.states = np.empty([2 ** n_lanes], np.complex128)        

    def get_n_lanes(self) :
        return self.n_lanes

    def reset_lane_states(self) :
        self.lane_states = [-1] * self.n_lanes
    
    def get_lane_state(self, lane) :
        return self.lane_states[lane]
    
    def set_lane_state(self, lane, value) :
        self.lane_states[lane] = value
    
    # internal methods
    
    def __getitem__(self, key) :
        return self.states[key]

    def __setitem__(self, key, value) :
        self.states[key] = value


class PyQubitProcessor :

    def reset(self) :
        pass
    
    def initialize_qubit_states(self, qstates, n_lanes) :
        qstates.allocate(n_lanes)
        
    def reset_qubit_states(self, qstates) :
        qstates.states[:] = np.complex128()
        qstates.states[0] = 1
        qstates.reset_lane_states()
        
    def calc_probability(self, qstates, local_lane) :

        bitmask_lane = 1 << local_lane
        bitmask_hi = ~((2 << local_lane) - 1)
        bitmask_lo = (1 << local_lane) - 1
        n_loops = 2 ** (qstates.get_n_lanes() - 1)
        prob = 0.
        for idx in range(n_loops) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            qs = qstates[idx_lo]
            prob += (qs * qs.conj()).real

        return prob
        
    def measure(self, rand_num, qstates, local_lane) :

        prob = self.calc_probability(qstates, local_lane)

        bitmask_lane = 1 << local_lane
        bitmask_hi = ~((2 << local_lane) - 1)
        bitmask_lo = (1 << local_lane) - 1
        n_loops = 2 ** (qstates.get_n_lanes() - 1)

        if (rand_num < prob) :
            value = 0
            norm = 1. / math.sqrt(prob)
            for idx in range(n_loops) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] *= norm
                qstates[idx_hi] = 0.
        else :
            value = 1
            norm = 1. / math.sqrt(1. - prob)
            for idx in range(n_loops) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] = 0.
                qstates[idx_hi] *= norm

        return value

    def apply_reset(self, qstates, local_lane) :

        bitmask_lane = 1 << local_lane
        bitmask_hi = ~((2 << local_lane) - 1)
        bitmask_lo = (1 << local_lane) - 1
        n_loops = 2 ** (qstates.get_n_lanes() - 1)

        for idx in range(n_loops) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            idx_hi = idx_lo | bitmask_lane
            qstates[idx_lo] = qstates[idx_hi]
            qstates[idx_hi] = 0.


    def apply_unary_gate(self, gate_type, _adjoint, qstates, local_lane) :
        mat = gate_type.pymat()
        if _adjoint :
            mat = adjoint(mat)
        
        bitmask_lane = 1 << local_lane
        bitmask_hi = ~((2 << local_lane) - 1)
        bitmask_lo = (1 << local_lane) - 1
        n_loops = 2 ** (qstates.get_n_lanes() - 1)
        for idx in range(n_loops) :
            idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
            idx_hi = idx_lo | bitmask_lane
            qs0 = qstates[idx_lo]
            qs1 = qstates[idx_hi]
            qsout = np.matmul(mat, np.array([qs0, qs1], np.complex128).T)
            qstates[idx_lo] = qsout[0]
            qstates[idx_hi] = qsout[1]

    def apply_control_gate(self, gate_type, _adjoint,
                           qstates, local_control_lanes, local_target_lane) :
        mat = gate_type.pymat()
        if _adjoint :
            mat = adjoint(mat)
            
        # create control bit mask
        bits = [1 << lane for lane in local_control_lanes]
        control_mask = 0
        for bit in bits : # bits stores control bits.
            control_mask |= bit
        
        # target bit
        target_bit = 1 << local_target_lane
        bits.append(target_bit)
        bits = sorted(bits)

        mask = bits[0] - 1
        masks = [ mask ]
        for idx in range(len(bits) - 1) :
            mask = (bits[idx + 1] - 1) & (~(bits[idx] * 2 - 1))
            masks.append(mask)
        mask = ~(bits[-1] * 2 - 1)
        masks.append(mask)

        n_loops = 1 << (qstates.get_n_lanes() - len(bits))
        n_shifts = len(bits) + 1
        for idx in range(n_loops) :
            idx_0 = 0
            for shift in range(n_shifts) :
                idx_0 |= (idx << shift) & masks[shift]
            idx_0 |= control_mask
            idx_1 = idx_0 | target_bit
 
            qs0 = qstates[idx_0]
            qs1 = qstates[idx_1]
            qsout = np.matmul(mat, np.array([qs0, qs1], np.complex128).T)
            qstates[idx_0] = qsout[0]
            qstates[idx_1] = qsout[1]

    def get_states(self, values, array_offset, mathop,
                   lanes, qstates_list, n_states, start, step) :
        arranged = []
        for qstates in qstates_list :
            lane_transform = [(1 << lane.external, 1 << lane.local)
                              for lane in lanes if lane.qstates == qstates]
            arranged.append((qstates, lane_transform))
        
        for idx in range(n_states) :
            val = 1.
            for qstates, lane_transform in arranged :
                ext_idx = start + step * idx
                # convert to local idx
                local_idx = 0
                for bit_pair in lane_transform :
                    if (bit_pair[0] & ext_idx) != 0 :
                        local_idx |= bit_pair[1]
                state = qstates[local_idx]
                val *= mathop(state)
                
            values[array_offset + idx] = val

def create_qubit_states(dtype) :
    return QubitStates()

def create_qubit_processor(dtype) :
    return PyQubitProcessor()
