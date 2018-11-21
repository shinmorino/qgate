import numpy as np
from . import glue

class NativeQubitStates :
    def __init__(self, ptr, qproc) :
        self.ptr = ptr
        self._qproc = qproc
        
    def __del__(self) :
        glue.qubit_states_delete(self.ptr)
        self.ops = None
        
    def allocate(self, qreglist) :
        qregids = [qreg.id for qreg in qreglist]
        glue.qubit_states_allocate(self.ptr, qregids)

    def deallocate(self) :
        glue.qubit_states_deallocate(self.ptr)

    def reset(self) :
        glue.qubit_states_reset(self.ptr)

    def get_n_qregs(self) :
        return glue.qubit_states_get_n_qregs(self.ptr)
