import qgate
import numpy as np
from qgate.script import *

import sys
this = sys.modules[__name__]

this.n_qubits = 32 # using 32 GB of device memory.

if False:
    # for testing with one GPU.
    this.n_qubits = 28
    # logically splitting one device to 4 devices with 2 GB of memory for each.
    qgate.simulator.cudaruntime.set_preference(device_ids = [ 0, 0, 0, 0 ],
                                               max_po2idx_per_chunk = 29,
                                               memory_store_size = (1 << 31))

def run(circuit, caption) :

    sim = qgate.simulator.cuda(dtype=np.float32, circuit_prep = qgate.prefs.static)
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
