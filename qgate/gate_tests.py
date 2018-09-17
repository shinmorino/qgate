import simulator

from qasm_model import *
from qasm_processor import *

def run(caption) :
    program = current_program()
    program = expand_register_lists(program)
    #circuit = map_to_circuit([program])
    seperated = seperate_programs(program)
    circuit = map_to_circuit(seperated)

    sim = simulator.py(circuit)
    sim.prepare()
    while sim.run_step() :
        pass

    print(caption)
    qstates = sim.get_qstates(0)
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

# Pauli gate
    
init_program()
qreg = allocate_qreg(2)
#x(qreg[0])
#x(qreg[1])
cx(qreg[0], qreg[1])
run('CNot gate')
