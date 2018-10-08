from . import simulator
from . import pyruntime
from . import cpuruntime
try :
    from . import cudaruntime
except :
    pass

def py(program) :
    sim = simulator.Simulator(pyruntime.PyRuntime())
    sim.set_program(program)
    return sim

def cpu(program) :
    sim = simulator.Simulator(cpuruntime.CPURuntime())
    sim.set_program(program)
    return sim

def cuda(program) :
    sim = simulator.Simulator(cudaruntime.CUDARuntime())
    sim.set_program(program)
    return sim
    
