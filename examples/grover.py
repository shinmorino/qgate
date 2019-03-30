# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import qgate
from qgate.script import *

# Glover's algorithm

# allocating qubit register
qregs = new_qregs(2)
q0, q1 = qregs[0], qregs[1]

# building a quantum circuit.
circuit = [
    [H(qreg) for qreg in qregs],
    H(q1),
    ctrl(q0).X(q1),
    H(q1),
    [H(qreg) for qreg in qregs],
    [X(qreg) for qreg in qregs],
    H(q1),
    ctrl(q0).X(q1),
    H(q1),
    [X(qreg) for qreg in qregs],
    [H(qreg) for qreg in qregs]
]

# measure
cregs = new_references(2)
circuit += [
    measure(cregs[0], qregs[0]),
    measure(cregs[1], qregs[1])
]

sim = qgate.simulator.py()
sim.run(circuit)

qgate.dump(sim.qubits)
qgate.dump(sim.values)

sim.terminate()
