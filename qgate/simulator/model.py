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
class CregArrayDict :
    def __init__(self, creg_arrays) :
        self.creg_array_dict = dict()
        for creg_array in creg_arrays :
            values = np.zeros([creg_array.length()])
            self.creg_array_dict[creg_array] = values

    def __getitem__(self, key) :
        return self.creg_array_dict[key]

    def dump(self) :
        for creg_array, creg_values in self.creg_array_dict.items() :
            print(creg_array)
            for idx, value in enumerate(creg_values) :
                print("{:d}:".format(idx), value)
    
    def set(self, creg, value) :
        values = self.creg_array_dict[creg.creg_array]
        values[creg.idx] = value
