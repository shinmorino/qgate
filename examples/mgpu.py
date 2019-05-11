import qgate
import numpy as np
from qgate.script import *

import sys
this = sys.modules[__name__]

this.mgpu = True
this.n_qubits = 28

def run(circuit, caption) :

    if this.mgpu :
        qgate.simulator.cudaruntime.set_preference(device_ids = [ 0, 0, 0, 0 ], max_po2idx_per_chunk = 29, memory_store_size = (1 << 31) - 10)

    sim = qgate.simulator.cuda(dtype=np.float32, circuit_prep = qgate.prefs.one_static)
    sim.run(circuit)

    print(caption)
    states = sim.qubits.states[:]
    
    print(sim.obs(creg))
    print()
    
    sim.terminate()

qregs = new_qregs(this.n_qubits)
creg = new_reference();
circuit = [
    X(qregs[0]),
    [ctrl(qregs[idx]).X(qregs[idx + 1]) for idx in range(this.n_qubits - 1)],
    [I(qreg) for qreg in qregs],
    measure(creg, qregs[-1])
]

run(circuit, 'tests')
