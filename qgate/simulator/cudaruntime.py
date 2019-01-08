from . import cudaext
import numpy as np
import weakref 
from .native_qubit_processor import NativeQubitProcessor
from .native_qubit_states import NativeQubitStates

import sys
this = sys.modules[__name__]
this.qubit_states = weakref.WeakValueDictionary()

def create_qubit_states(dtype, processor) :
    ptr = cudaext.qubit_states_new(dtype)
    qstates = NativeQubitStates(ptr, processor)
    this.qubit_states[id(qstates)] = qstates
    return qstates

def create_qubit_processor(dtype) :
    qproc = NativeQubitProcessor(dtype, cudaext.qubit_processor_new(dtype))
    return qproc

def module_finalize() :
    qstates = this.qubit_states.values()
    for qs in qstates :
        qs.deallocate()
    cudaext.module_finalize()

import atexit
atexit.register(module_finalize)
