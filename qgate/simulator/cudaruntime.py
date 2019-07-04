try :
    from . import cudaext
except :
    import sys
    if sys.version_info[0] == 2 :
        del cudaext
    raise
        
import numpy as np
import weakref 
from .native_qubit_processor import NativeQubitProcessor
from .native_qubit_states import NativeQubitStates
from .native_sampling_pool import NativeSamplingPool
from . import glue

import sys
this = sys.modules[__name__]

# initialization flag.
this.initialized = False

# dictionary that holds native instances.
this.native_instances = weakref.WeakValueDictionary()

def set_preference(device_ids = [], max_po2idx_per_chunk = -1, memory_store_size = -1) :
    if this.initialized :
        raise RuntimeError('already initialized.')
    this.max_po2idx_per_chunk = max_po2idx_per_chunk
    this.device_ids = device_ids
    this.memory_store_size = memory_store_size

def set_preference(device_ids = [], max_po2idx_per_chunk = -1, memory_store_size = -1) :
    if this.initialized :
        raise RuntimeError('already initialized.')
    if len(device_ids) != 0 :
        this.device_ids = device_ids
    if max_po2idx_per_chunk != -1 :
        this.max_po2idx_per_chunk = max_po2idx_per_chunk
    if memory_store_size != -1 :
        this.memory_store_size = memory_store_size

def reset_preference(device_ids = [], max_po2idx_per_chunk = -1, memory_store_size = -1) :
    this.device_ids = []
    this.max_po2idx_per_chunk = -1
    this.memory_store_size = -1

def create_qubit_states(dtype) :
    if not this.initialized :
        module_init()
    ptr = cudaext.qubit_states_new(dtype)
    qstates = NativeQubitStates(ptr)
    this.native_instances[id(qstates)] = qstates
    return qstates

def create_qubit_processor(dtype) :
    if not this.initialized :
        module_init();
    qproc = CUDAQubitProcessor(dtype, cudaext.qubit_processor_new(dtype))
    this.native_instances[id(qproc)] = qproc
    return qproc

class CUDAQubitProcessor(NativeQubitProcessor) :
    def __init__(self, dtype, ptr) :
        NativeQubitProcessor.__init__(self, dtype, ptr)

    def create_sampling_pool(self, qreg_ordering,
                             n_lanes, n_hidden_lanes, lane_trans, empty_lanes,
                             sampling_pool_factory = None) :
        return self._create_sampling_pool(qreg_ordering, n_lanes, n_hidden_lanes, lane_trans,
                                          empty_lanes, True, sampling_pool_factory)

def module_init() :
    cudaext.devices_initialize(this.device_ids, this.max_po2idx_per_chunk, this.memory_store_size)
    this.initialized = True

def module_finalize() :
    instances = this.native_instances.values()
    for ptr in instances :
        ptr.delete()
    if this.initialized :
        cudaext.devices_clear()
    this.initialized = False

import atexit
atexit.register(module_finalize)

# set default preference
this.reset_preference()

