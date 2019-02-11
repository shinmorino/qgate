import numpy as np
from . import glue

class NativeQubitStates :
    def __init__(self, ptr) :
        self.ptr = ptr
        
    def __del__(self) :
        self.delete()

    def delete(self) :
        if hasattr(self, 'ptr') :
            glue.qubit_states_delete(self.ptr)
            del self.ptr

    def get_n_lanes(self) :
        return glue.qubit_states_get_n_lanes(self.ptr)
