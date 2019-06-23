import qgate
from qgate.script import *
import numpy as np

n_qregs = 10
qregs = new_qregs(n_qregs)
empty = new_qregs(3)

cregs = new_references(n_qregs)

circuit = [
    [H(qreg) for qreg in qregs],
    [X(qreg) for qreg in qregs],
    [ctrl(qregs[idx]).X(qregs[idx + 1]) for idx in range(0, n_qregs - 1)],
]

# qgate.dump(circuit)

sim = qgate.simulator.cuda()
sim.run(circuit)

if False :
    # get reduced prob array.
    class DummySamplingPool :
        def __init__(self, prob, empty_lanes, qreg_ordering) :
            self.prob = prob
            self.empty_lanes = empty_lanes
            self.qreg_ordering = qreg_ordering

    spool = sim.qubits.create_sampling_pool(qregs[2:], DummySamplingPool)
    print(len(spool.prob))

np.random.seed(0)

spool = sim.qubits.create_sampling_pool(qregs[2:])
obslist = spool.sample(1024)
print(obslist)

print('histgram:')
print(obslist.histgram())

#qgate.dump(obslist)
#qgate.dump(obslist.histgram())
