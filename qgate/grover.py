# https://www.ibm.com/developerworks/jp/cloud/library/cl-quantum-computing/index.html
import simulator

from qasm_model import *
from qasm_processor import *

# Glover's algorithm

# allocating qubit register
qregs = allocate_qreg(2)
q0, q1 = qregs[0], qregs[1]

# applying gates
h(qregs)
h(q1)
cx(q0, q1)
h(q1)
h(qregs)
x(qregs)
h(q1)
cx(q0, q1)
h(q1)
x(qregs)
h(qregs)

# measure
cregs = allocate_creg(2)
measure(qregs, cregs)

program = current_program()
program = expand_register_lists(program)
seperated = seperate_programs(program)

sim = simulator.py(seperated)
sim.prepare()
while sim.run_step() :
    pass
sim.terminate()

qstates = sim.get_qstates(0)
qstates.dump()


# engine = simulator.py(program)

# engine.run(program)

#engine.fit()
#engine.optimize()

#for circuit in engine.circuits :
#    runner = engine.create_runner(circuit)
#    runner.start()
#    while runner.run_step() :
#        pass
#    runner.end()


#engine.run()
# equivalent to the following lines
#
# engine.start()
# while engine.run_step() :
#   pass
# engine.end()
