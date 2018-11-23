from . import cudaext
import numpy as np
from .native_qubit_processor import NativeQubitProcessor
from .native_qubit_states import NativeQubitStates

def create_qubit_states(dtype) :
    ptr = cudaext.qubit_states_new(dtype)
    proc = get_processor(dtype)
    return NativeQubitStates(ptr, proc)

def get_processor(dtype) :
    if dtype == np.float64 :
        return this.processor_fp64
    elif dtype == np.float32 :
        return this.processor_fp32
    else :
        raise RuntimeError('dtype not supported.')

# module-level initialization
import sys
this = sys.modules[__name__]
this.processor_fp32 = NativeQubitProcessor(cudaext.qubit_processor_new(np.float32))
this.processor_fp64 = NativeQubitProcessor(cudaext.qubit_processor_new(np.float64))

def module_finalize() :
    cudaext.module_finalize()

import atexit
atexit.register(module_finalize)
