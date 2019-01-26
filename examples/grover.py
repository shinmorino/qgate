# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import qgate
from qgate.script import *
from qgate.script.qelib1 import *

# Glover's algorithm

# allocating qubit register
qregs = allocate_qregs(2)
q0, q1 = qregs[0], qregs[1]

circuit = new_circuit()

# applying gates
circuit.add(
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
cregs = allocate_cregs(2)
circuit.add(
    measure(qregs, cregs)
)

circuit = process(circuit, isolate_circuits = False)

sim = qgate.simulator.py(circuit)
sim.prepare()
sim.run()

qubits = sim.qubits()
qgate.dump(qubits)
cregdict = sim.get_cregdict()
qgate.dump_creg_values(cregdict)

sim.terminate()
