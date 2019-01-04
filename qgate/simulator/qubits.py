from . import glue
import numpy as np


# math operations
def null(v) :
    return v
def abs2(v) :
    return v.real ** 2 + v.imag ** 2
prob = abs2


def qproc(qstates) :
    """ get qubit processor instance associated with qubit states. """
    return qstates._qproc

class Qubits :
    def __init__(self, dtype) :
        self.dtype = dtype
        self.qstates_dict = {}

    def __del__(self) :
        self.qstates_dict.clear()

    def get_n_qubits(self) :
        n_qubits = 0
        for qstates in self.qstates_dict.values() :
            n_qubits += qstates.get_n_qregs()
        return n_qubits
    
    def add_qubit_states(self, key, qstates) :
        self.qstates_dict[key] = qstates
    
    def __getitem__(self, key) :
        return self.qstates_dict[key]

    def get_qubit_states(self) :
        return self.qstates_dict.values()

    def calc_probability(self, qreg) :
        prob = 1.
        for qstates in self.qstates_dict.values() :
            if qstates.has_qreg(qreg) :
                proc = qproc(qstates)
                prob *= proc.calc_probability(qstates, qreg)
        return prob
    
    def get_states(self, mathop = None) :
        if mathop is None :
            mathop = null
            
        n_states = 1 << self.get_n_qubits()
        if mathop == null :
            dtype = np.complex64 if self.dtype == np.float32 else np.complex128
        elif mathop == abs2 :
            dtype = self.dtype
            
        all_qstates = dict()    
        for qstates in self.qstates_dict.values() :
            proc = qproc(qstates)
            if not proc in all_qstates :
                all_qstates[proc] = [qstates]
            else :
                all_qstates[proc].append(qstates)

        values = np.empty([n_states], dtype)
        for proc, qstates_list in all_qstates.items() :
            proc.get_states(values, mathop, qstates_list)
        return values
