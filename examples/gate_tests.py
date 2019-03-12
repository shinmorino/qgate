from __future__ import print_function
import qgate
from qgate.script import *


def run(circuit, caption) :
    prefs = {'isolate_circuits' : True}
    sim = qgate.simulator.py(**prefs)
#    sim = qgate.simulator.cpu(**prefs)
#    sim = qgate.simulator.cuda(**prefs)
    sim.run(circuit)

    print(caption)
    qgate.dump(sim.qubits, qgate.simulator.prob)
    qgate.dump(sim.qubits)
    qgate.dump(sim.values)
    print()
    
    sim.terminate()

# initial

circuit = new_circuit()
qreg = new_qreg()
circuit.add(a(qreg))
run(circuit, 'initial')

# Hadamard gate
circuit = new_circuit()
qreg = new_qreg()
circuit.add(h(qreg))
run(circuit, 'Hadamard gate')


# Pauli gate
    
circuit = new_circuit()
qreg = new_qreg()
circuit.add(x(qreg))
run(circuit, 'Pauli gate')


# reset
    
circuit = new_circuit()
qreg = new_qreg()
valueref = new_reference()  # test new_reference()
circuit.add(x(qreg),
            measure(qreg, valueref),
            reset(qreg))
run(circuit, 'reset')


# CX gate
    
circuit = new_circuit()
qregs = new_qregs(2)
circuit.add(x(qregs[0]),
            x(qregs[1]),
            cntr(qregs[0]).x(qregs[1])
)
run(circuit, 'CX gate')


# 2 seperated flows

circuit = new_circuit()
qreg = new_qregs(2)
circuit.add(x(qreg[0]),
            x(qreg[1]))
run(circuit, '2 seperated flows')

# measure
circuit = new_circuit()
qregs = new_qregs(2)
refs = new_references(2)
circuit.add(
    [x(qreg) for qreg in qregs],
    measure(qregs[0], refs[0]),
    measure(qregs[1], refs[1]),
)
run(circuit, 'measure')

# if clause
circuit = new_circuit()
qreg = new_qregs(2)
ref = new_reference()
circuit.add(x(qreg[0]),
            measure(qreg[0], ref),
            if_(ref, 1, x(qreg[1]))
)
run(circuit, "if clause")

# exp gate

# expia, expiz
circuit = new_circuit()
qreg = new_qregs(1)
#circuit.add(expia(0)(qreg[0]), expiz(0)(qreg[0]))
circuit.add(expia(0)(qreg[0]))
run(circuit, "single qubit exp gate")

