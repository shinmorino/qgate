from . import simulator
from . import pymatrix   # python matrix definitions.
from . import cpumatrix  # cpu matrix definitions.
from . import pyruntime
from . import cpuruntime
from . import utils
import numpy as np

try :
    from . import cudaruntime
except :
    pass


from .qubits import null, abs2, prob


def py(**prefs) :
    sim = simulator.Simulator(pyruntime, **prefs)
    return sim

def cpu(**prefs) :
    sim = simulator.Simulator(cpuruntime, **prefs)
    return sim

def cuda(**prefs) :
    sim = simulator.Simulator(cudaruntime, **prefs)
    return sim
    
