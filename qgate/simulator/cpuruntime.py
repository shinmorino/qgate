from . import cpuext
import numpy as np


class Qubits :
    def __init__(self) :
        self.ptr = cpuext.qubits_new()
        self.qstates_dict = {}

    def __del__(self) :
        cpuext.qubits_detach_qubit_states(self.ptr)
        self.qstates_dict.clear()
        cpuext.qubits_delete(self.ptr)

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
        cpuext.qubits_add_qubit_states(self.ptr, key, qstates.ptr)

    def prepare(self) :
        pass

    def get_states(self) :
        n_states = 1 << self.get_n_qubits()
        states = np.empty([n_states], np.complex64)
        cpuext.qubits_get_states(self.ptr, states, 0, n_states, 0)
        return states

    def get_probabilities(self) :
        n_states = 1 << self.get_n_qubits()
        probs = np.empty([n_states], np.float32)
        cpuext.qubits_get_probabilities(self.ptr, probs, 0, n_states, 0)
        return probs


class QubitStates :
    def __init__(self, qreglist) :
        qreg_id_list = [qreg.id for qreg in qreglist]
        self.ptr = cpuext.qubit_states_new(qreg_id_list)

    def __del__(self) :
        cpuext.qubit_states_delete(self.ptr)

    def get_n_qregs(self) :
        return cpuext.qubit_states_get_n_qregs(self.ptr)
    
    
def measure(rand_num, qstates, qreg) :
    return cpuext.measure(rand_num, qstates.ptr, qreg.id)
    
def apply_reset(qstates, qreg) :
    cpuext.apply_reset(qstates.ptr, qreg.id)
                                
def apply_unary_gate(mat, qstates, qreg) :
    mat = np.asarray(mat, dtype=np.complex64, order='C')
    cpuext.apply_unary_gate(mat, qstates.ptr, qreg.id)

def apply_control_gate(mat, qstates, control, target) :
    mat = np.asarray(mat, dtype=np.complex64, order='C')
    cpuext.apply_control_gate(mat, qstates.ptr, control.id, target.id)
