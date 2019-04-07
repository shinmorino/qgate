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
        
    def measure(self, rand_num, qstates, local_lane) :
        return glue.qubit_processor_measure(self.ptr, rand_num, qstates.ptr, local_lane)
    
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
                   lanes, qstates_list, n_states, start, step) :
        if mathop == qubits.null :
            mathop = 0
        elif mathop == qubits.abs2 :
            mathop = 1
        else :
            raise RuntimeError('unknown math operation, {}'.format(str(mathop)))

        if len(qstates_list) == 0 :
            if n_states != 1 or start != 0 :
                raise RuntimeError('cannot set values.')
            values[0] = 1.
            return

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
                                        lane_transform_list, qstates_ptrs, len(lanes),
                                        n_states, start, step)
