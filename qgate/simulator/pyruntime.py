import numpy as np
import math
from . import qubits
from . import observation

def adjoint(mat) :
    return np.conjugate(mat.T)

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
        qstates.reset_lane_states()
        
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

    def join(self, qstates, qstates_list, n_new_qregs) :
        it = iter(qstates_list)
        qs = next(it, None)
        vec = qs.states

        while True :
            qs = next(it, None)
            if qs is None :
                break
            vec = np.kron(vec, qs.states)

        len_vec = len(vec)
        qstates.states[:len_vec] = vec[:]
        qstates.states[len_vec:] = 0.

    def decohere(self, value, prob, qstates, local_lane) :

        bitmask_lane = 1 << local_lane
        bitmask_hi = ~((2 << local_lane) - 1)
        bitmask_lo = (1 << local_lane) - 1
        n_loops = 2 ** (qstates.get_n_lanes() - 1)

        if (value == 0) :
            norm = 1. / math.sqrt(prob)
            for idx in range(n_loops) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] *= norm
                qstates[idx_hi] = 0.
        else :  # value == 1
            norm = 1. / math.sqrt(1. - prob)
            for idx in range(n_loops) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates[idx_lo] = 0.
                qstates[idx_hi] *= norm

        return value
    
    def decohere_and_separate(self, value, prob, qstates0, qstates1, qstates, local_lane) :

        bitmask_lane = 1 << local_lane
        bitmask_hi = ~((2 << local_lane) - 1)
        bitmask_lo = (1 << local_lane) - 1
        n_loops = 2 ** (qstates.get_n_lanes() - 1)

        if value == 0 :
            norm = 1. / math.sqrt(prob)
            for idx in range(n_loops) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                qstates0[idx] = norm * qstates[idx_lo]
            qstates1[0] = 1.
            qstates1[1] = 0.
        else :
            norm = 1. / math.sqrt(1. - prob)
            for idx in range(n_loops) :
                idx_lo = ((idx << 1) & bitmask_hi) | (idx & bitmask_lo)
                idx_hi = idx_lo | bitmask_lane
                qstates0[idx] = norm * qstates[idx_hi]
            qstates1[0] = 0.
            qstates1[1] = 1.

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
                   lane_trans, empty_lanes, n_states, start, step) :

        empty_lane_mask = 0
        for empty_lane_pos in empty_lanes :
            empty_lane_mask |= 1 << empty_lane_pos
        
        arranged = []
        for qstates, lanes in lane_trans :
            lanebit_list = list()
            for lane in lanes :
                lanebit_list.append((1 << lane.external, 1 << lane.local))
            arranged.append((qstates, lanebit_list))

        for idx in range(n_states) :
            if idx & empty_lane_mask != 0 :
                val = 0.
            else :
                val = 1.
                for qstates, lanebit_list in arranged :
                    ext_idx = start + step * idx
                    # convert to local idx
                    local_idx = 0
                    for lane_bit in lanebit_list :
                        if (lane_bit[0] & ext_idx) != 0 :
                            local_idx |= lane_bit[1]
                    state = qstates[local_idx]
                    val *= mathop(state)
                
            values[array_offset + idx] = val

    def create_sampling_pool(self, qreg_ordering,
                             n_lanes, n_hidden_lanes, lane_trans, empty_lanes,
                             sampling_pool_factory = None) :

        if sampling_pool_factory is None :
            sampling_pool_factory = PySamplingPool

        if n_hidden_lanes == 0 :
            n_states = 1 << n_lanes
            probs = np.empty([n_states], np.float64)
            self.get_states(probs, 0, qubits.abs2, lane_trans, [], n_states, 0, 1)
            return sampling_pool_factory(probs, empty_lanes, qreg_ordering)

        # reorder external lane (-1, n_prob)
        all_lanes = list()
        hidden_idx = 0
        for qs, lanelist in lane_trans :
            for lane in lanelist :
                if lane.external == -1 :
                    lane.external = hidden_idx
                    hidden_idx += 1
                else :
                    lane.external += n_hidden_lanes

        n_states = 1 << (n_lanes + n_hidden_lanes)
        probs = np.empty([n_states], np.float64)
        self.get_states(probs, 0, qubits.abs2, lane_trans, [], n_states, 0, 1)

        n_probs = 1 << n_lanes
        reduced = np.sum(probs.reshape(n_probs, -1), 1)

        return sampling_pool_factory(reduced, empty_lanes, qreg_ordering)


class PySamplingPool :
    def __init__(self, prob, empty_lanes, qreg_ordering) :
        self.cumprob = np.cumsum(prob)
        norm = 1. / self.cumprob[-1]
        self.cumprob *= norm

        self.qreg_ordering = qreg_ordering
        self.empty_lanes = empty_lanes
        self.mask = 0
        for idx in empty_lanes :
            self.mask |= 1 << idx

    def sample(self, n_samples, randnum = None) :
        # norm = 1. / cprobs[-1]
        # cprobs *= norm
        obs = np.empty([n_samples], dtype = np.int64)
        if randnum is None :
            randnum = np.random.random_sample([n_samples])
        obs = np.searchsorted(self.cumprob, randnum, side = 'right')
        if self.mask != 0 :
            self.shift_for_empty_lanes(obs)

        obslist = observation.ObservationList(self.qreg_ordering, obs, self.mask)
        return obslist

    def shift_for_empty_lanes(self, obs) :
        # create control bit mask
        bits = [1 << lane for lane in self.empty_lanes]
        n_shifts = len(bits) + 1

        mask = bits[0] - 1
        masks = [ mask ]
        for idx in range(len(bits) - 1) :
            mask = (bits[idx + 1] - 1) & (~(bits[idx] * 2 - 1))
            masks.append(mask)
        mask = ~(bits[-1] * 2 - 1)
        masks.append(mask)

        for idx in range(len(obs)) :
            v = obs[idx]
            v_shifted = 0
            for shift in range(n_shifts) :
                v_shifted |= (v << shift) & masks[shift]
            obs[idx] = v_shifted


def create_qubit_states(dtype) :
    return QubitStates()

def create_qubit_processor(dtype) :
    return PyQubitProcessor()
