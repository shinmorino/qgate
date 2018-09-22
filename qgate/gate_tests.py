from qasm.qelib1 import *
from qasm.script import *
from qasm.processor import *

import simulator.simulator

def run(caption) :
    program = current_program()
    program = process(program, seperate_circuit = False)
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

init_program()
qreg = allocate_qreg(1)
run('initial')

# Hadamard gate
    
init_program()
qreg = allocate_qreg(1)
h(qreg)
run('Hadamard gate')


# Pauli gate
    
init_program()
qreg = allocate_qreg(1)
x(qreg)
run('Pauli gate')

# CNot gate
    
init_program()
qreg = allocate_qreg(2)
#x(qreg[0])
#x(qreg[1])
cx(qreg[0], qreg[1])
run('CNot gate')


# 2 seperated flows

init_program()
qreg = allocate_qreg(2)
#x(qreg[0])
x(qreg[1])
run('2 seperated flows')
