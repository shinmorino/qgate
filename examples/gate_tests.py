from __future__ import print_function
import qgate
from qgate.script import *
import math

def run(circuit, caption) :
    prefs = {'isolate_circuits' : True}
#    sim = qgate.simulator.py(**prefs)
    sim = qgate.simulator.cpu(**prefs)
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
circuit.add(A(qreg))
run(circuit, 'initial')

# Hadamard gate
circuit = new_circuit()
qreg = new_qreg()
circuit.add(H(qreg))
run(circuit, 'Hadamard gate')


# Pauli gate
    
circuit = new_circuit()
qreg = new_qreg()
circuit.add(X(qreg))
run(circuit, 'Pauli gate')


# reset
    
circuit = new_circuit()
qreg = new_qreg()
valueref = new_reference()  # test new_reference()
circuit.add(X(qreg),
            measure(valueref, qreg),
            reset(qreg))
run(circuit, 'reset')


# CX gate
    
circuit = new_circuit()
qregs = new_qregs(2)
circuit.add(X(qregs[0]),
            X(qregs[1]),
            ctrl(qregs[0]).X(qregs[1])
)
run(circuit, 'CX gate')


# 2 seperated flows

circuit = new_circuit()
qregs = new_qregs(2)
circuit.add(X(qregs[0]),
            X(qregs[1]))
run(circuit, '2 seperated flows')

# measure
circuit = new_circuit()
qregs = new_qregs(2)
refs = new_references(2)
circuit.add(
    [X(qregs) for qregs in qregs],
    measure(refs[0], qregs[0]),
    measure(refs[1], qregs[1])
)
run(circuit, 'measure')

# if clause
circuit = new_circuit()
qregs = new_qregs(2)
ref = new_reference()
circuit.add(X(qregs[0]),
            measure(ref, qregs[0]),
            if_(ref, 1)(X(qregs[1]))
)
run(circuit, "if clause")

# exp gate

# expia, expiz
circuit = new_circuit()
qregs = new_qregs(1)
#circuit.add(expia(0)(qregs[0]), expiz(0)(qregs[0]))
circuit.add(Expia(0)(qregs[0]))
run(circuit, "single qubit exp gate")

circuit = new_circuit()
qregs = new_qregs(4)
circuit.add(Expi(math.pi / 8)(X(qregs[0]), Y(qregs[1]), Z(qregs[2]), A(qregs[3])))
run(circuit, "exp gate")

# pauli measure
circuit = new_circuit()
qregs = new_qregs(4)
circuit.add(measure(ref, (X(qregs[0]), Y(qregs[1]), Z(qregs[2]), A(qregs[3]) ) ))
run(circuit, "pmeasure")

# prob
circuit = new_circuit()
qregs = new_qregs(4)
circuit.add([H(qregs) for qregs in qregs],
            prob(ref, qregs[0]))
run(circuit, "prob")

# pauli prob
circuit = new_circuit()
qregs = new_qregs(4)
circuit.add(prob(ref, (X(qregs[0]), Y(qregs[1]), Z(qregs[2]), A(qregs[3]) ) ))
run(circuit, "pauli prob")
