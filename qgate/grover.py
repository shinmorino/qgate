# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import simulator

from qasm.script import *
from qasm.qelib1 import *
import qasm.processor

init_program()

# Glover's algorithm

# allocating qubit register
qregs = allocate_qreg(2)
q0, q1 = qregs[0], qregs[1]

# applying gates
h(qregs)
h(q1)
cx(q0, q1)
h(q1)
h(qregs)
x(qregs)
h(q1)
cx(q0, q1)
h(q1)
x(qregs)
h(qregs)

# measure
cregs = allocate_creg(2)
measure(qregs, cregs)

program = current_program()
program = qasm.processor.process(program, seperate_circuit = True)

sim = simulator.py(program)
sim.prepare()
while sim.run_step() :
    pass

qstates = sim.get_qstates(0)
qstates.dump()
cregs = sim.get_cregs()
cregs.dump()

sim.terminate()
