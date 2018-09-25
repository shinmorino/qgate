from qasm.qelib1 import *
from qasm.script import *
from qasm.processor import *

import simulator.simulator
from simulator.utils import dump_qubit_states, dump_creg_values


def run(caption) :
    program = current_program()
    program = process(program, isolate_circuits = True)
    sim = simulator.py(program)
    
    sim.prepare()
    while sim.run_step() :
        pass

    print(caption)
    problist = sim.get_probability_list()
    dump_qubit_states(problist)
    creg_dict = sim.get_creg_dict()
    dump_creg_values(creg_dict)
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

# measure
new_program()
qreg = allocate_qreg(2)
creg = allocate_creg(2)
op(
    a(qreg),
    measure(qreg, creg)
)
run('measure')

# if clause
new_program()
qreg = allocate_qreg(2)
creg = allocate_creg(1)
op(x(qreg[0]),
   a(qreg),
   measure(qreg[0], creg[0]),
   if_c(creg, 0, x(qreg[1]))
)
run("if clause")
