from . import cpuext

def create_qubit_states(dtype) :
    ptr = cpufactory.qubit_states_new(dtype)
    proc = get_processor(dtype)
    return QubitStates(ptr, qop)

def get_processor(dtype) :
    if dtype == np.float64 :
        return this.qubit_ops_fp64
    elif dtype == np.float32 :
        return this.qubit_ops_fp32
    else :
        raise RuntimeError('dtype not supported.')

# module-level initialization
import sys
this = sys.modules[__name__]
this.processor_fp32 = cpufactory.processor_new(np.float32)
this.processor_fp64 = cpufactory.processor_new(np.float64)
