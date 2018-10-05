# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import qgate
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

new_program()

# Glover's algorithm

# allocating qubit register
qregs = allocate_qreg(2)
q0, q1 = qregs[0], qregs[1]

# applying gates
op(
    h(qregs),
    h(q1),
    cx(q0, q1),
    h(q1),
   h(qregs),
   x(qregs),
   h(q1),
   cx(q0, q1),
   h(q1),
   x(qregs),
   h(qregs)
)

# measure
cregs = allocate_creg(2)
op(measure(qregs, cregs))

program = current_program()
program = qgate.model.processor.process(program, isolate_circuits = False)

sim = qgate.simulator.py(program)
sim.prepare()
sim.run()

qubits = sim.get_qubits()
qgate.dump_probabilities(qubits)
creg_dict = sim.get_creg_dict()
qgate.dump_creg_values(creg_dict)

sim.terminate()
