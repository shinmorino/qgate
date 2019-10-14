from __future__ import print_function
import qgate
from qgate.script import *
import math

def run(caption, circuit, refs = None) :
    prefs = { qgate.prefs.circuit_prep: qgate.prefs.dynamic }
#    sim = qgate.simulator.py(**prefs)
    sim = qgate.simulator.cpu(**prefs)
#    sim = qgate.simulator.cuda(**prefs)
    sim.run(circuit)

    print(caption)
    qgate.dump(sim.qubits)
    qgate.dump(sim.qubits.prob)
    if refs is not None :
        print('observation: {}'.format(sim.obs(refs)))
    print()
    
    sim.terminate()

# initial

qreg = new_qreg()
circuit = I(qreg)
run('initial', circuit)

# Hadamard gate
qreg = new_qreg()
circuit = H(qreg)
run('Hadamard gate', circuit)


# Pauli gate
    
qreg = new_qreg()
circuit = X(qreg)
run('Pauli gate', circuit)

# reset
    
qreg = new_qreg()
valueref = new_reference()  # test new_reference()
circuit = [X(qreg),
           measure(valueref, qreg),
           reset(qreg)]
run('reset', circuit, valueref)


# CX gate
    
qregs = new_qregs(2)
circuit = [X(qregs[0]),
           X(qregs[1]),
           ctrl(qregs[0]).X(qregs[1])]
run('CX gate', circuit)


# 2 separated flows

qregs = new_qregs(2)
circuit = [X(qregs[0]), X(qregs[1])]
run('2 separated flows', circuit)

# measure
qregs = new_qregs(2)
refs = new_references(2)
circuit = [
    [X(qreg) for qreg in qregs],
    measure(refs[0], qregs[0]),
    measure(refs[1], qregs[1])
]
run('measure', circuit, refs)

# if clause
qregs = new_qregs(2)
ref = new_reference()
circuit = [
    X(qregs[0]),
    measure(ref, qregs[0]),
    if_(ref, 1, X(qregs[1]))
]
run('if clause', circuit, ref)

# exp gate

# expii, expiz
qregs = new_qregs(1)
circuit = [Expii(0)(qregs[0]), Expiz(0)(qregs[0])]
run('single qubit exp gate', circuit)

qregs = new_qregs(4)
circuit = Expi(math.pi / 8)(X(qregs[0]), Y(qregs[1]), Z(qregs[2]), I(qregs[3]))
run('exp gate', circuit)

# pauli measure
qregs = new_qregs(4)
circuit = measure(ref, [X(qregs[0]), Y(qregs[1]), Z(qregs[2]), I(qregs[3])])
run('pmeasure', circuit)

# prob
qregs = new_qregs(4)
circuit = [
    [H(qreg) for qreg in qregs],
    prob(ref, qregs[0])
]
run('prob', circuit)

# pauli prob
qregs = new_qregs(4)
circuit = prob(ref, [X(qregs[0]), Y(qregs[1]), Z(qregs[2]), I(qregs[3])])
run('pauli prob', circuit)

qregs = new_qregs(2)
ref = new_reference()
circuit = [ctrl(qregs[0]).X(qregs[1]),
           measure(ref, qregs[1]),
           release_qreg(qregs[1])]
run('remove qreg', circuit)
