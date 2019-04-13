import qgate
import numpy as np
from qgate.script import *

import sys
this = sys.modules[__name__]

this.mgpu = False
this.n_qubits = 28

def run(circuit, caption) :
    sim = qgate.simulator.cuda(dtype=np.float32, isolate_circuits = False)
    
    n_lanes_per_device = -1
    device_ids = []

    if this.mgpu :
        n_lanes_per_device = this.n_qubits - 1
        device_ids = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    sim.run(circuit)

    print(caption)
    states = sim.qubits.get_states()
    
    qgate.dump(sim.values)
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
