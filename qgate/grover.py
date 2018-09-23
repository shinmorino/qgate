# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import simulator

from qasm.script import *
from qasm.qelib1 import *
import qasm.processor

new_program()

# Glover's algorithm

# allocating qubit register
qregs = allocate_qreg(2)
q0, q1 = qregs[0], qregs[1]

# applying gates
op(h(qregs),
   h(q1),
   cx(q0, q1),
   h(q1),
   h(qregs),
   x(qregs),
   h(q1),
   cx(q0, q1),
   h(q1),
   x(qregs),
   h(qregs))

# measure
cregs = allocate_creg(2)
op(measure(qregs, cregs))

program = current_program()
program = qasm.processor.process(program, isolate_circuits = True)

sim = simulator.py(program)
sim.prepare()
while sim.run_step() :
    pass

qstates = sim.get_qstates(0)
qstates.dump()
creg_array_dict = sim.get_creg_array_dict()
creg_array_dict.dump()

sim.terminate()
