from qasm.qelib1 import *
from qasm.script import *
from qasm.processor import *

import simulator.simulator

def run(caption) :
    program = current_program()
    program = process(program, seperate_circuit = True)
    sim = simulator.py(program)
    
    sim.prepare()
    while sim.run_step() :
        pass

    print(caption)
    for i in range(sim.get_n_circuits()) :
        qstates = sim.get_qstates(i)
        qstates.dump()
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
op(x(qreg))
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
