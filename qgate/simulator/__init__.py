from . import simulator
from . import pyruntime
from . import cpuruntime
import numpy as np

try :
    from . import cudaruntime
except :
    pass


from .qubits import null, abs2, prob


def py(program, dtype = np.float64) :
    sim = simulator.Simulator(pyruntime, dtype)
    sim.set_program(program)
    return sim

def cpu(program, dtype = np.float64) :
    sim = simulator.Simulator(cpuruntime, dtype)
    sim.set_program(program)
    return sim

def cuda(program, dtype = np.float64) :
    sim = simulator.Simulator(cudaruntime)
    sim.set_program(program)
    return sim
    
