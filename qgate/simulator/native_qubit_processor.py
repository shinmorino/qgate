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
        
    def initialize_qubit_states(self, qregs, qstates) :
        qregids = [qreg.id for qreg in qregs]
        glue.qubit_processor_initialize_qubit_states(self.ptr, qregids, qstates.ptr)

    def reset_qubit_states(self, qstates) :
        glue.qubit_processor_reset_qubit_states(self.ptr, qstates.ptr)
        
    def measure(self, rand_num, qstates, qreg) :
        return glue.qubit_processor_measure(self.ptr, rand_num, qstates.ptr, qreg.id)
    
    def apply_reset(self, qstates, qreg) :
        glue.qubit_processor_apply_reset(self.ptr, qstates.ptr, qreg.id)

    def apply_unary_gate(self, mat, qstates, qreg) :
        mat = np.asarray(mat, dtype=np.complex128, order='C')
        glue.qubit_processor_apply_unary_gate(self.ptr, mat, qstates.ptr, qreg.id)

    def apply_control_gate(self, mat, qstates, control, target) :
        mat = np.asarray(mat, dtype=np.complex128, order='C')
        glue.qubit_processor_apply_control_gate(self.ptr, mat, qstates.ptr, control.id, target.id)

    def get_states(self, values, mathop, qstates_list) :
        if mathop == qubits.null :
            mathop = 0
        elif mathop == qubits.abs2 :
            mathop = 1
        else :
            raise RuntimeError('unknown math operation, {}'.format(str(mathop)))

        qstates_ptrs = [qstates.ptr for qstates in qstates_list]
        glue.qubit_processor_get_states(self.ptr,
                                        values, 0,
                                        mathop,
                                        qstates_ptrs,
                                        0,
                                        values.shape[0])
