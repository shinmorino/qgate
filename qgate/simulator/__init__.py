from . import simulator
from . import pyruntime
from . import cpuruntime

def py(program) :
    sim = simulator.Simulator(pyruntime.PyRuntime())
    sim.set_program(program)
    return sim

def cpu(program) :
    sim = simulator.Simulator(cpuruntime.CPURuntime())
    sim.set_program(program)
    return sim
    
