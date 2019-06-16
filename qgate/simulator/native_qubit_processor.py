import numpy as np
from . import qubits
from . import glue
from . import native_sampling_pool

class NativeQubitProcessor :

    def __init__(self, dtype, ptr) :
        self.dtype = dtype
        self.ptr = ptr
        
    def __del__(self) :
        self.delete()

    def delete(self) :
        if hasattr(self, 'ptr') :
            glue.qubit_processor_delete(self.ptr)
            del self.ptr
        
    def reset(self) :
        glue.qubit_processor_reset(self.ptr)
        
    def initialize_qubit_states(self, qstates, n_lanes) :
        glue.qubit_processor_initialize_qubit_states(self.ptr, qstates.ptr, n_lanes)
        qstates.reset_lane_states()

    def reset_qubit_states(self, qstates) :
        glue.qubit_processor_reset_qubit_states(self.ptr, qstates.ptr)
        qstates.reset_lane_states()
        
    def calc_probability(self, qstates, local_lane) :
        return glue.qubit_processor_calc_probability(self.ptr, qstates.ptr, local_lane)
    
    def join(self, qstates, qstates_list, n_new_qregs) :
        qstates_ptrs = [qs.ptr for qs in qstates_list]
        return glue.qubit_processor_join(self.ptr, qstates.ptr, qstates_ptrs, n_new_qregs)
    
    def decohere(self, value, prob, qstates, local_lane) :
        return glue.qubit_processor_decohere(self.ptr, value, prob, qstates.ptr, local_lane)
    
    def decohere_and_separate(self, value, prob, qstates0, qstates1, qstates, local_lane) :
        return glue.qubit_processor_decohere_and_separate(self.ptr, value, prob, qstates0.ptr, qstates1.ptr, qstates.ptr, local_lane)
    
    def apply_reset(self, qstates, local_lane) :
        glue.qubit_processor_apply_reset(self.ptr, qstates.ptr, local_lane)

    def apply_unary_gate(self, gate_type, _adjoint, qstates, local_lane) :
        assert hasattr(gate_type, 'cmatf'), \
            'gate type, {}, does not have cmatf attribute.'.format(repr(gate_type))
        glue.qubit_processor_apply_unary_gate(self.ptr, gate_type, _adjoint,
                                              qstates.ptr, local_lane)

    def apply_control_gate(self, gate_type, _adjoint,
                           qstates, local_control_lanes, local_target_lane) :
        glue.qubit_processor_apply_control_gate(self.ptr, gate_type, _adjoint, qstates.ptr,
                                                local_control_lanes, local_target_lane)

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

        empty_lane_mask = NativeQubitProcessor.create_lane_mask(empty_lanes)

        qstates_ptrs, lanepos_array_list = NativeQubitProcessor.translate_transform(lane_trans)
        glue.qubit_processor_get_states(self.ptr,
                                        values, offset,
                                        mathop,
                                        lanepos_array_list, empty_lane_mask, qstates_ptrs, n_qregs,
                                        n_states, start, step)

    def create_sampling_pool(self, qreg_ordering,
                             n_lanes, n_hidden_lanes, lane_trans, empty_lanes,
                             sampling_pool_factory = None) :
        # reorder external lane
        all_lanes = list()
        hidden_idx = 0
        for qs, lanelist in lane_trans :
            for lane in lanelist :
                if lane.external == -1 :
                    lane.external = hidden_idx
                    hidden_idx += 1
                else :
                    lane.external += n_hidden_lanes

        n_states = 1 << n_lanes
        qstates_ptrs, lanepos_array_list = NativeQubitProcessor.translate_transform(lane_trans)
        if sampling_pool_factory is not None :
            prob = np.empty([n_states], self.dtype)
            ptr = glue.qubit_processor_prepare_prob_array(self.ptr, prob,
                        lanepos_array_list, qstates_ptrs, n_lanes, n_hidden_lanes)
            return sampling_pool_factory(prob, empty_lanes, qreg_ordering)

        ptr = glue.qubit_processor_create_sampling_pool(self.ptr, lanepos_array_list, qstates_ptrs,
                                                        n_lanes, n_hidden_lanes, empty_lanes)

        empty_lane_mask = NativeQubitProcessor.create_lane_mask(empty_lanes)
        return native_sampling_pool.NativeSamplingPool(ptr, qreg_ordering, empty_lane_mask)

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
