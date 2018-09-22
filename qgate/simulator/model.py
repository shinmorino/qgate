import numpy as np
import math

# representing a single qubit or entangled qbits.
class QubitStates :
    def __init__(self, n_qubits) :
        self.states = np.zeros([2 ** n_qubits], np.complex128)
        self.states[0] = 1

    def __getitem__(self, key) :
        return self.states[key]

    def __setitem__(self, key, value) :
        self.states[key] = value

    def dump(self) :
        for idx, state in enumerate(self.states) :
            print('{0:08b}'.format(idx), (state * state.conj()).real)

    
# representing a single qubit or entangled qbits.
class Cregs :
    def __init__(self, n_cregs) :
        self.values = np.zeros([n_cregs], np.int32)

    def __getitem__(self, key) :
        return self.values[key]

    def __setitem__(self, key, value) :
        self.values[key] = value

    def dump(self) :
        for idx, value in enumerate(self.values) :
            print("{:d}:".format(idx), value)
