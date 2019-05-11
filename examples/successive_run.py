import qgate
from qgate.script import *

# creating simulator instance
sim = qgate.simulator.cpu()

# creating quantum registers that represent qubits.
qregs = new_qregs(3)
qreg0, qreg1, qreg2 = qregs

print('1 qubit')
sim.run(H(qreg0))
qgate.dump(sim.qubits)

print('2 qubits')
sim.run(X(qreg1))
qgate.dump(sim.qubits)

print('3 qubits')
sim.run(H(qreg2))
qgate.dump(sim.qubits)

print('measure and release qreg0')
refs = new_references(3)
sim.run([measure(refs[0], qreg0),
         release_qreg(qreg0)])
qgate.dump(sim.qubits)

print('measure and release qreg1')
sim.run([measure(refs[1], qreg1),
         release_qreg(qreg1)])
qgate.dump(sim.qubits)

print('measure and release qreg2')
sim.run([measure(refs[2], qreg2),
         release_qreg(qreg2)])

qgate.dump(sim.qubits)
obs = sim.obs(refs)
print('observaion: {}'.format(obs))
