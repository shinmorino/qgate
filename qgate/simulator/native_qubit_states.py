import numpy as np
from . import glue

class NativeQubitStates :
    def __init__(self, ptr, qproc) :
        self.ptr = ptr
        self._qproc = qproc
        
    def __del__(self) :
        glue.qubit_states_delete(self.ptr)

    def get_n_qregs(self) :
        return glue.qubit_states_get_n_qregs(self.ptr)
