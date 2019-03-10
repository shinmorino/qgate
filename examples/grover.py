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
    [h(qreg) for qreg in qregs],
    h(q1),
    cntr(q0).x(q1),
    h(q1),
    [h(qreg) for qreg in qregs],
    [x(qreg) for qreg in qregs],
    h(q1),
    cntr(q0).x(q1),
    h(q1),
    [x(qreg) for qreg in qregs],
    [h(qreg) for qreg in qregs]
)

# measure
cregs = new_references(2)
circuit.add(
    measure(qregs[0], cregs[0]),
    measure(qregs[1], cregs[1])
)

sim = qgate.simulator.py()
sim.run(circuit)

qgate.dump(sim.qubits)
qgate.dump(sim.values)

sim.terminate()
