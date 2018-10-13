from . import cudaext
import numpy as np
import math


class Qubits :
    def __init__(self) :
        self.ptr = cudaext.qubits_new()
        self.qstates_dict = {}

    def __del__(self) :
        cudaext.qubits_detach_qubit_states(self.ptr)
        self.qstates_dict.clear()
        cudaext.qubits_delete(self.ptr)

    def get_n_qubits(self) :
        n_qubits = 0
        for qstates in self.qstates_dict.values() :
            n_qubits += qstates.get_n_qregs()
        return n_qubits

    def __getitem__(self, key) :
        return self.qstates_dict[key]
        
    def allocate_qubit_states(self, key, qreglist) :
        qstates = QubitStates(qreglist)
        self.qstates_dict[key] = qstates
        cudaext.qubits_add_qubit_states(self.ptr, key, qstates.ptr)
        
    def prepare(self) :
        cudaext.qubits_prepare(self.ptr)
        
    def get_states(self) :
        n_states = 1 << self.get_n_qubits()
        states = np.empty([n_states], np.complex64)
        cudaext.qubits_get_states(self.ptr, states, 0, n_states, 0)
        return states

    def get_probabilities(self) :
        n_states = 1 << self.get_n_qubits()
        probs = np.empty([n_states], np.float32)
        cudaext.qubits_get_probabilities(self.ptr, probs, 0, n_states, 0)
        return probs


class QubitStates :
    def __init__(self, qreglist) :
        qreg_id_list = [qreg.id for qreg in qreglist]
        self.ptr = cudaext.qubit_states_new(qreg_id_list)

    def __del__(self) :
        cudaext.qubit_states_delete(self.ptr)

    def get_n_qregs(self) :
        return cudaext.qubit_states_get_n_qregs(self.ptr)
    
    
def measure(rand_num, qstates, qreg) :
    return cudaext.measure(rand_num, qstates.ptr, qreg.id)
    
def apply_reset(qstates, qreg) :
    cudaext.apply_reset(qstates.ptr, qreg.id)
                                
def apply_unary_gate(mat, qstates, qreg) :
    mat = np.asarray(mat, dtype=np.complex64, order='C')
    cudaext.apply_unary_gate(mat, qstates.ptr, qreg.id)

def apply_control_gate(mat, qstates, control, target) :
    mat = np.asarray(mat, dtype=np.complex64, order='C')
    cudaext.apply_control_gate(mat, qstates.ptr, control.id, target.id)



def module_finalize() :
    cudaext.module_finalize()

import atexit
atexit.register(module_finalize)
