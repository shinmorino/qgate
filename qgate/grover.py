# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
from qasm.script import *
from qasm.qelib1 import *
import qasm.processor

import simulator
from simulator.utils import dump_qubit_states

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

problist = sim.get_probability_list()
dump_qubit_states(problist)
creg_array_dict = sim.get_creg_arrays()
creg_array_dict.dump()

sim.terminate()
