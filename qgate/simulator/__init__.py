from . import simulator
from . import pymatrix   # python matrix definitions.
from . import cpumatrix  # cpu matrix definitions.
from . import pyruntime
from . import cpuruntime
import numpy as np

try :
    from . import cudaruntime
except :
    pass


from .qubits import null, abs2, prob


def py(circuits, dtype = np.float64) :
    sim = simulator.Simulator(pyruntime, dtype)
    sim.set_circuits(circuits)
    return sim

def cpu(circuits, dtype = np.float64) :
    sim = simulator.Simulator(cpuruntime, dtype)
    sim.set_circuits(circuits)
    return sim

def cuda(circuits, dtype = np.float64) :
    sim = simulator.Simulator(cudaruntime, dtype)
    sim.set_circuits(circuits)
    return sim
    
