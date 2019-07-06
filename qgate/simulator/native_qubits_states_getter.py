from . import qubits
from . import glue
from . import native_sampling_pool
import numpy as np

class NativeQubitsStatesGetter :
    def __init__(self, dtype, ptr) :
        self.dtype = dtype
        self.ptr = ptr
        
    def __del__(self) :
        self.delete()

    def delete(self) :
        if hasattr(self, 'ptr') :
            glue.qubits_states_getter_delete(self.ptr)
            del self.ptr

    def get_states(self, values, offset, mathop,
                   lane_trans, empty_lanes, n_states, start, step) :
        if mathop == qubits.null :
            mathop = 0
        elif mathop == qubits.abs2 :
            mathop = 1
        else :
            raise RuntimeError('unknown math operation, {}'.format(str(mathop)))

        n_lanes = sum([len(lanepos_list) for qstates, lanepos_list in lane_trans])
        n_qregs = n_lanes + len(empty_lanes)

        empty_lane_mask = NativeQubitsStatesGetter.create_lane_mask(empty_lanes)

        qstates_ptrs, lanepos_array_list = NativeQubitsStatesGetter.translate_transform(lane_trans)
        glue.qubits_states_getter_get_states(self.ptr, values, offset, mathop,
                                lanepos_array_list, empty_lane_mask, qstates_ptrs, n_qregs,
                                n_states, start, step)
        
    def _create_sampling_pool(self, qreg_ordering,
                              n_lanes, n_hidden_lanes, lane_trans, empty_lanes,
                              hidden_lane_in_msb,
                              sampling_pool_factory = None) :

        if hidden_lane_in_msb :
            NativeQubitsStatesGetter.place_hidden_lanes_in_msb(lane_trans, n_lanes, n_hidden_lanes)
        else :
            NativeQubitsStatesGetter.place_hidden_lanes_in_lsb(lane_trans, n_lanes, n_hidden_lanes)
        
        n_states = 1 << n_lanes
        qstates_ptrs, lanepos_array_list = NativeQubitsStatesGetter.translate_transform(lane_trans)
        if sampling_pool_factory is not None :
            prob = np.empty([n_states], self.dtype)
            ptr = glue.qubits_states_getter_prepare_prob_array(self.ptr, prob,
                            lanepos_array_list, qstates_ptrs, n_lanes, n_hidden_lanes)
            return sampling_pool_factory(prob, empty_lanes, qreg_ordering)

        ptr = glue.qubits_states_getter_create_sampling_pool(self.ptr,
                                                             lanepos_array_list, qstates_ptrs,
                                                             n_lanes, n_hidden_lanes,
                                                             empty_lanes)
        
        empty_lane_mask = NativeQubitsStatesGetter.create_lane_mask(empty_lanes)
        return native_sampling_pool.NativeSamplingPool(ptr, qreg_ordering, empty_lane_mask)

    @staticmethod
    def place_hidden_lanes_in_msb(lane_trans, n_lanes, n_hidden_lanes) :
        hidden_idx = n_lanes
        for qs, lanelist in lane_trans :
            for lane in lanelist :
                if lane.external == -1 :
                    lane.external = hidden_idx
                    hidden_idx += 1

    @staticmethod
    def place_hidden_lanes_in_lsb(lane_trans, n_lanes, n_hidden_lanes) :
        hidden_idx = 0
        for qs, lanelist in lane_trans :
            for lane in lanelist :
                if lane.external == -1 :
                    lane.external = hidden_idx
                    hidden_idx += 1
                else :
                    lane.external += n_hidden_lanes

    @staticmethod
    def translate_transform(lane_trans) :
        qstates_ptrs = list()
        lanepos_array_list = list()
        for qstates, lanepos_list in lane_trans :
            qstates_ptrs.append(qstates.ptr)
            lanepos_array = [None] * len(lanepos_list)
            for lanepos in lanepos_list :
                # lane_transform[local_lane] -> external_lane
                lanepos_array[lanepos.local] = lanepos.external
            lanepos_array_list.append(lanepos_array)
        return qstates_ptrs, lanepos_array_list

    @staticmethod
    def create_lane_mask(lanepos_list) :
        mask = 0
        for pos in lanepos_list :
            mask |= 1 << pos
        return mask
