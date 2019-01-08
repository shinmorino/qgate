import numpy as np
from . import glue

class NativeQubitStates :
    def __init__(self, ptr, qproc) :
        self.ptr = ptr
        self._qproc = qproc
        
    def __del__(self) :
        self.deallocate()

    def deallocate(self) :
        if hasattr(self, 'ptr') :
            glue.qubit_states_delete(self.ptr)
            del self.ptr

    def get_n_qregs(self) :
        return glue.qubit_states_get_n_qregs(self.ptr)

    def has_qreg(self, qreg) :
        if not hasattr(self, '_qregs') :
            return False
        return qreg in self._qregs
