# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import qgate
from qgate.script import *

# Glover's algorithm

# allocating qubit register
qregs = new_qregs(2)
q0, q1 = qregs[0], qregs[1]

circuit = new_circuit()

# applying gates
circuit.add(
    h(qregs),
    h(q1),
    cntr(q0).x(q0),
    h(q1),
    h(qregs),
    x(qregs),
    h(q1),
    cntr(q0).x(q1),
    h(q1),
    x(qregs),
    h(qregs)
)

# measure
cregs = new_references(2)
circuit.add(
    measure(qregs[0], cregs[0]),
    measure(qregs[1], cregs[1])
)

circuit = process(circuit, isolate_circuits = False)

sim = qgate.simulator.py(circuit)
sim.prepare()
sim.run()

qgate.dump(sim.qubits)
qgate.dump(sim.values)

sim.terminate()
