import numpy as np
from . import qubits
from . import glue


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
                   lanes, empty_lanes, qstates_list, n_states, start, step) :
        if mathop == qubits.null :
            mathop = 0
        elif mathop == qubits.abs2 :
            mathop = 1
        else :
            raise RuntimeError('unknown math operation, {}'.format(str(mathop)))

        n_qregs = len(lanes) + len(empty_lanes)

        empty_lane_mask = 0
        for empty_lane_pos in empty_lanes :
            empty_lane_mask |= 1 << empty_lane_pos

        lane_transform_list = []
        for qstates in qstates_list :
            lanes_in_qstates = [lane for lane in lanes if lane.qstates == qstates]
            # lane_transform[local_lane] -> external_lane
            lane_transform = [None] * len(lanes_in_qstates)
            for lane in lanes_in_qstates :
                lane_transform[lane.local] = lane.external
            lane_transform_list.append(lane_transform)
        
        qstates_ptrs = [qstates.ptr for qstates in qstates_list]
        glue.qubit_processor_get_states(self.ptr,
                                        values, offset,
                                        mathop,
                                        lane_transform_list, empty_lane_mask, qstates_ptrs, n_qregs,
                                        n_states, start, step)
