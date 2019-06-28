from . import cpuext
import numpy as np
from .native_qubit_processor import NativeQubitProcessor
from .native_qubit_states import NativeQubitStates

def create_qubit_states(dtype) :
    ptr = cpuext.qubit_states_new(dtype)
    return NativeQubitStates(ptr)

def create_qubit_processor(dtype) :
    return CPUQubitProcessor(dtype, cpuext.qubit_processor_new(dtype))

class CPUQubitProcessor(NativeQubitProcessor) :
    def __init__(self, dtype, ptr) :
        NativeQubitProcessor.__init__(self, dtype, ptr)

    def create_sampling_pool(self, qreg_ordering,
                             n_lanes, n_hidden_lanes, lane_trans, empty_lanes,
                             sampling_pool_factory = None) :
        return self._create_sampling_pool(qreg_ordering, n_lanes, n_hidden_lanes, lane_trans,
                                          empty_lanes, False, sampling_pool_factory)
