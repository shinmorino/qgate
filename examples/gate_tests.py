from __future__ import print_function
import qgate
from qgate.qasm.qelib1 import *
from qgate.qasm.script import *


def run(caption) :
    program = current_program()
    program = qgate.model.process(program, isolate_circuits = True)
#    sim = qgate.simulator.py(program)
    sim = qgate.simulator.cpu(program)
#    sim = qgate.simulator.cuda(program)
    
    sim.prepare()
    while sim.run_step() :
        pass

    print(caption)
    qubits = sim.get_qubits()
    qgate.dump(qubits, qgate.simulator.prob)
    qgate.dump(qubits)
    creg_dict = sim.get_creg_dict()
    qgate.dump_creg_values(creg_dict)
    print()
    
    sim.terminate()

# initial

new_program()
qreg = allocate_qreg(1)
run('initial')
fin_program()

# Hadamard gate
new_program()
qreg = allocate_qreg(1)
op(h(qreg))
run('Hadamard gate')


# Pauli gate
    
new_program()
qreg = allocate_qreg(1)
op(x(qreg))
run('Pauli gate')


# reset
    
new_program()
qreg = allocate_qreg(1)
creg = allocate_creg(1)
op(x(qreg))
op(measure(qreg, creg))
op(reset(qreg))
run('reset')


# CX gate
    
new_program()
qreg = allocate_qreg(2)
op(x(qreg[0]),
   x(qreg[1]),
   cx(qreg[0], qreg[1]))
run('CX gate')


# 2 seperated flows

new_program()
qreg = allocate_qreg(2)
op(x(qreg[0]),
   x(qreg[1]))
run('2 seperated flows')

# measure
new_program()
qreg = allocate_qreg(2)
creg = allocate_creg(2)
op(
    x(qreg),
    measure(qreg, creg)
)
run('measure')

# if clause
new_program()
qreg = allocate_qreg(2)
creg = allocate_creg(1)
op(x(qreg[0]),
   measure(qreg[0], creg[0]),
   if_(creg, 1, x(qreg[1]))
)
run("if clause")
