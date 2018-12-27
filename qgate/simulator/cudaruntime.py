from . import cudaext
import numpy as np
from .native_qubit_processor import NativeQubitProcessor
from .native_qubit_states import NativeQubitStates

def create_qubit_states(dtype, processor) :
    ptr = cudaext.qubit_states_new(dtype)
    return NativeQubitStates(ptr, processor)

def create_qubit_processor(dtype) :
    return NativeQubitProcessor(dtype, cudaext.qubit_processor_new(dtype))

def module_finalize() :
    cudaext.module_finalize()

import atexit
atexit.register(module_finalize)
