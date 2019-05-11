import qgate
from qgate.script import *

# creating a quantum register that represent a qubit like OpenQASM.
qreg = new_qreg()

# creating a quantum circuit with H gate.
circuit = [H(qreg)]

# creating reference as a placeholder to store measured result.
ref = new_reference()

# measure qreg
circuit.append(measure(ref, qreg))

# creating simulator instance
sim = qgate.simulator.py()

# running the prepared circuit on sim.
sim.run(circuit)

# showing a measured value.
value = sim.obs(ref)
print('Measured value: {}'.format(value))
