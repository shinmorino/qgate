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

    def has_qreg(self, qreg_id) :
        if not hasattr(self, 'qreg_id_list') :
            return False
        return qreg_id in self.qreg_id_list
