import qgate
from qgate.script import *

n_qregs = 10
qregs = new_qregs(n_qregs)
cregs = new_references(n_qregs)

circuit = [
    [H(qreg) for qreg in qregs],
    [X(qreg) for qreg in qregs],
    [ctrl(qregs[idx]).X(qregs[idx + 1]) for idx in range(0, n_qregs - 1)],
    [measure(c, q) for c, q in zip(cregs, qregs)]
]

# qgate.dump(circuit)

sim = qgate.simulator.cpu(circuit_prep = qgate.prefs.dynamic)
sim.run(circuit)
print('\nobserved: {}'.format(sim.obs(cregs)))

obslist = sim.sample(circuit, cregs, 512)
hist = obslist.histgram()
# print(obslist)
print('histgram:')
print(hist)

#qgate.dump(obslist)
#qgate.dump(obslist.histgram())
