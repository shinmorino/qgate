import numpy as np

# representing a single qubit or entangled qbits.
class QubitStates :
    def __init__(self, n_qubits) :
        self.states = np.array([2 ** n_qubits], np.complex128)

