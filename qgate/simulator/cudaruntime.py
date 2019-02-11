from . import cudaext
import numpy as np
import weakref 
from .native_qubit_processor import NativeQubitProcessor
from .native_qubit_states import NativeQubitStates

import sys
this = sys.modules[__name__]
this.native_instances = weakref.WeakValueDictionary()

def create_qubit_states(dtype) :
    ptr = cudaext.qubit_states_new(dtype)
    qstates = NativeQubitStates(ptr)
    this.native_instances[id(qstates)] = qstates
    return qstates

def create_qubit_processor(dtype) :
    qproc = NativeQubitProcessor(dtype, cudaext.qubit_processor_new(dtype))
    this.native_instances[id(qproc)] = qproc
    return qproc

def module_finalize() :
    instances = this.native_instances.values()
    for ptr in instances :
        ptr.delete()
    cudaext.module_finalize()

import atexit
atexit.register(module_finalize)
