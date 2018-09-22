import numpy as np
import math

# representing a single qubit or entangled qbits.
class QubitStates :
    def __init__(self, qregset) :
        self.states = np.zeros([2 ** len(qregset)], np.complex128)
        self.states[0] = 1
        self.qregset = qregset

    def __getitem__(self, key) :
        return self.states[key]

    def __setitem__(self, key, value) :
        self.states[key] = value

    def get_n_lanes(self) :
        return len(self.qregset)

    def get_lane(self, qreg) :
        return list(self.qregset).index(qreg)
        
    def dump(self) :
        for idx, state in enumerate(self.states) :
            print('{0:08b}'.format(idx), (state * state.conj()).real)

    
# representing a single qubit or entangled qbits.
class Cregs :
    def __init__(self, cregset) :
        self.values = np.zeros([len(cregset)], np.int32)
        self.cregset = cregset

    def __getitem__(self, key) :
        return self.values[key]

    def __setitem__(self, key, value) :
        self.values[key] = value

    def dump(self) :
        for idx, value in enumerate(self.values) :
            print("{:d}:".format(idx), value)

    def set_cregset(self, cregmap) :
        self.cregset = cregset
    
    def get_idx(self, creg) :
        return list(self.cregset).index(creg)

