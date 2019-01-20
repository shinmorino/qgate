import numpy as np
from . import qubits
from . import glue

class NativeQubitProcessor :

    def __init__(self, dtype, ptr) :
        self.dtype = dtype
        self.ptr = ptr

    def clear(self) :
        glue.qubit_processor_clear(self.ptr)

    def prepare(self) :
        glue.qubit_processor_prepare(self.ptr)
        
    def initialize_qubit_states(self, qregset, qstates, n_lanes_per_chunk, device_ids) :
        qreg_id_list = [qreg.id for qreg in qregset]
        glue.qubit_processor_initialize_qubit_states(self.ptr, qreg_id_list, qstates.ptr, n_lanes_per_chunk, device_ids)
        qstates.qreg_id_list = qreg_id_list

    def reset_qubit_states(self, qstates) :
        glue.qubit_processor_reset_qubit_states(self.ptr, qstates.ptr)
        
    def calc_probability(self, qstates, qreg_id) :
        return glue.qubit_processor_calc_probability(self.ptr, qstates.ptr, qreg_id)
        
    def measure(self, rand_num, qstates, qreg_id) :
        return glue.qubit_processor_measure(self.ptr, rand_num, qstates.ptr, qreg_id)
    
    def apply_reset(self, qstates, qreg_id) :
        glue.qubit_processor_apply_reset(self.ptr, qstates.ptr, qreg_id)

    def apply_unary_gate(self, mat, qstates, qreg_id) :
        mat = np.asarray(mat, dtype=np.complex128, order='C')
        glue.qubit_processor_apply_unary_gate(self.ptr, mat, qstates.ptr, qreg_id)

    def apply_control_gate(self, mat, qstates, control_id, target_id) :
        mat = np.asarray(mat, dtype=np.complex128, order='C')
        glue.qubit_processor_apply_control_gate(self.ptr, mat, qstates.ptr, control_id, target_id)

    def get_states(self, values, offset, mathop,
                   qstates_list, n_qreg_lanes, n_states, start, step) :
        if mathop == qubits.null :
            mathop = 0
        elif mathop == qubits.abs2 :
            mathop = 1
        else :
            raise RuntimeError('unknown math operation, {}'.format(str(mathop)))

        qstates_ptrs = [qstates.ptr for qstates in qstates_list]
        glue.qubit_processor_get_states(self.ptr,
                                        values, offset,
                                        mathop,
                                        qstates_ptrs, n_qreg_lanes,
                                        n_states, start, step)
