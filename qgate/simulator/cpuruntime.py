import cpuext
import numpy as np
import math
import qasm.model as qasm
import random


class Qubits :
    def __init__(self, ptr, n_qubits) :
        self.ptr = ptr
        self.n_qubits = n_qubits

    def get_probabilities(self) :
        n_states = 1 << self.n_qubits
        probs = np.empty([n_states], np.float32)
        self.get_probabilities_internal(probs, 0, n_states, 0)
        return probs
        
    def get_probabilities_internal(self, probList, beginIdx, endIdx, list_offset) :
        cpuext.qubits_get_probabilities(self.ptr, probList, beginIdx, endIdx, list_offset)


class CPUKernel :

    def __init__(self) :
        self.ptr = cpuext.kernel_new()

    def __del__(self) :
        cpuext.kernel_delete(self.ptr)

    def set_qreglist(self, qreglist) :
        self.qreglist = qreglist
        qreg_id_list = [qreg.id for qreg in qreglist]
        cpuext.kernel_set_all_qreg_ids(self.ptr, qreg_id_list)
        
    # public methods
    
    def set_circuit(self, circuit_idx, circuit) :
        qreg_id_list = [qreg.id for qreg in circuit.qregs]
        cpuext.kernel_allocate_qubit_states(self.ptr, circuit_idx, qreg_id_list)

    def get_qubits(self) :
        qubit_ext = cpuext.kernel_get_qubits(self.ptr)
        return Qubits(qubit_ext, len(self.qreglist))
    
    # private methods
    
    def measure(self, rand_num, circ_idx, qreg) :
        return cpuext.kernel_measure(self.ptr, rand_num, circ_idx, qreg.id)
    
    def apply_reset(self, circ_idx, qreg) :
        cpuext.kernel_apply_reset(self.ptr, circ_idx, qreg.id)
                                
    def apply_unary_gate(self, mat, circ_idx, qreg) :
        mat = np.asarray(mat, dtype=np.complex64, order='C')
        cpuext.kernel_apply_unary_gate(self.ptr, mat, circ_idx, qreg.id)

    def apply_control_gate(self, mat, circ_idx, control, target) :
        mat = np.asarray(mat, dtype=np.complex64, order='C')
        cpuext.kernel_apply_control_gate(self.ptr, mat, circ_idx, control.id, target.id)
