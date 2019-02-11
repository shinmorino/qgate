import qgate
import numpy as np
from qgate.script import *

this.mgpu = False
this.n_qubits = 28

def run(circuit, caption) :
    circuit = qgate.model.process(circuit, isolate_circuits = False)
#    sim = qgate.simulator.py(program)
#    sim = qgate.simulator.cpu(program)
    sim = qgate.simulator.cuda(circuit, np.float32)

    n_lanes_per_device = -1
    device_ids = []

    if this.mgpu :
        n_lanes_per_device = this.n_qubits - 1
        device_ids = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    sim.prepare(n_lanes_per_device, device_ids)
    while sim.run_step() :
        pass

    print(caption)
    states = sim.qubits.get_states()
    
    qgate.dump(sim.values)
    print()
    
    sim.terminate()

circuit = new_circuit()

qregs = new_qregs(this.n_qubits)
creg = new_reference();
circuit.add(
    x(qregs[0]),
    [cntr(qregs[idx]).x(qregs[idx + 1]) for idx in range(this.n_qubits - 1)],
    [a(qreg) for qreg in qregs],
    measure(qregs[-1], creg)
)

run(circuit, 'tests')
