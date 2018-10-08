from . import cpuext
import numpy as np
import math


class Qubits :
    def __init__(self, n_qubits) :
        self.ptr = cpuext.qubits_new()
        self.n_qubits = n_qubits

    def __del__(self) :
        cpuext.qubits_delete(self.ptr)

    def get_n_qubits(self) :
        return self.n_qubits

    def get_states(self) :
        n_states = 1 << self.n_qubits
        states = np.empty([n_states], np.complex64)
        cpuext.qubits_get_states(self.ptr, states, 0, n_states, 0)
        return states

    def get_probabilities(self) :
        n_states = 1 << self.n_qubits
        probs = np.empty([n_states], np.float32)
        cpuext.qubits_get_probabilities(self.ptr, probs, 0, n_states, 0)
        return probs


class CPURuntime :

    def __init__(self) :
        self.ptr = cpuext.runtime_new()

    def __del__(self) :
        cpuext.runtime_delete(self.ptr)
        self.qubits = None

    def set_qreglist(self, qreglist) :
        self.qreglist = qreglist
        qreg_id_list = [qreg.id for qreg in qreglist]
        self.qubits = Qubits(len(qreglist))
        cpuext.runtime_set_qubits(self.ptr, self.qubits.ptr)
        cpuext.runtime_set_all_qreg_ids(self.ptr, qreg_id_list)
        
    # public methods
    
    def set_circuit(self, circuit_idx, circuit) :
        qreg_id_list = [qreg.id for qreg in circuit.qregs]
        cpuext.runtime_allocate_qubit_states(self.ptr, circuit_idx, qreg_id_list)

    def get_qubits(self) :
        return self.qubits
    
    # private methods
    
    def measure(self, rand_num, circ_idx, qreg) :
        return cpuext.runtime_measure(self.ptr, rand_num, circ_idx, qreg.id)
    
    def apply_reset(self, circ_idx, qreg) :
        cpuext.runtime_apply_reset(self.ptr, circ_idx, qreg.id)
                                
    def apply_unary_gate(self, mat, circ_idx, qreg) :
        mat = np.asarray(mat, dtype=np.complex64, order='C')
        cpuext.runtime_apply_unary_gate(self.ptr, mat, circ_idx, qreg.id)

    def apply_control_gate(self, mat, circ_idx, control, target) :
        mat = np.asarray(mat, dtype=np.complex64, order='C')
        cpuext.runtime_apply_control_gate(self.ptr, mat, circ_idx, control.id, target.id)
