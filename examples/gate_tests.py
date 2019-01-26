from __future__ import print_function
import qgate
from qgate.script import *
from qgate.script.qelib1 import *


def run(circuit, caption) :
    circuit = process(circuit, isolate_circuits = True)
    sim = qgate.simulator.py(circuit)
#    sim = qgate.simulator.cpu(circuit)
#    sim = qgate.simulator.cuda(circuit)
    
    sim.prepare()
    while sim.run_step() :
        pass

    print(caption)
    qubits = sim.qubits()
    qgate.dump(qubits, qgate.simulator.prob)
    qgate.dump(qubits)
    cregdict = sim.creg_values()
    qgate.dump_creg_values(cregdict)
    print()
    
    sim.terminate()

# initial

circuit = new_circuit()
qreg = allocate_qreg()
circuit.add([qreg])
run(circuit, 'initial')

# Hadamard gate
circuit = new_circuit()
qreg = allocate_qreg()
circuit.add(h(qreg))
run(circuit, 'Hadamard gate')


# Pauli gate
    
circuit = new_circuit()
qreg = allocate_qreg()
circuit.add(x(qreg))
run(circuit, 'Pauli gate')


# reset
    
circuit = new_circuit()
qreg = allocate_qreg()
creg = allocate_cregs(1)
circuit.add(x(qreg),
            measure(qreg, creg),
            reset(qreg))
run(circuit, 'reset')


# CX gate
    
circuit = new_circuit()
qregs = allocate_qregs(2)
circuit.add(x(qregs[0]),
            x(qregs[1]),
            cx(qregs[0], qregs[1]))
run(circuit, 'CX gate')


# 2 seperated flows

circuit = new_circuit()
qreg = allocate_qregs(2)
circuit.add(x(qreg[0]),
            x(qreg[1]))
run(circuit, '2 seperated flows')

# measure
circuit = new_circuit()
qreg = allocate_qregs(2)
creg = allocate_cregs(2)
circuit.add(
    x(qreg),
    measure(qreg, creg)
)
run(circuit, 'measure')

# if clause
circuit = new_circuit()
qreg = allocate_qregs(2)
creg = allocate_cregs(1)
circuit.add(x(qreg[0]),
            measure(qreg[0], creg[0]),
            if_(creg, 1, x(qreg[1]))
)
run(circuit, "if clause")
