from . import simulator
from . import pykernel
from . import cpukernel
from .utils import dump_probabilities, dump_creg_values

def py(program) :
    sim = simulator.Simulator(pykernel.PyKernel())
    sim.set_program(program)
    return sim

def cpu(program) :
    sim = simulator.Simulator(cpukernel.CPUKernel())
    sim.set_program(program)
    return sim
