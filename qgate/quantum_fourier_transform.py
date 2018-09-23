# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[4];
# creg c[4];
# x q[0]
# x q[2];
# barrier q;
# h q[0];
# cu1(pi/2) q[1],q[0];
# h q[1];
# cu1(pi/4) q[2],q[0];
# cu1(pi/2) q[2],q[1];
# h q[2];
# cu1(pi/8) q[3],q[0];
# cu1(pi/4) q[3],q[1];
# cu1(pi/2) q[3],q[2];
# h q[3];
# measure q -> c;


from qasm.script import *

# include "qelib1.inc";
from qasm.qelib1 import *

from qasm.processor import *
import simulator.simulator

new_program()

# // optional post-rotation for state tomography
# gate post q { }
def post(qregs) :
    return a(qregs)



q = allocate_qreg(4) # qreg q[4];
c = allocate_creg(4) # creg c[4];
op(
    x(q[0]),                       # x q[0]
    x(q[2]),                       # x q[2];
    barrier(q),                    # barrier q;
    h(q[0]),                       # h q[0];
    cu1(math.pi / 2., q[1], q[0]), # cu1(pi/2) q[1],q[0];
    h(q[1]),                       # h q[1];
    cu1(math.pi / 4., q[2], q[0]), # cu1(pi/4) q[2],q[0];
    cu1(math.pi / 2., q[2], q[1]), # cu1(pi/2) q[2],q[1];
    h(q[2]),                       # h q[2];
    cu1(math.pi / 8., q[3], q[0]), # cu1(pi/8) q[3],q[0];
    cu1(math.pi / 4., q[3], q[1]), # cu1(pi/4) q[3],q[1];
    cu1(math.pi / 2., q[3], q[2]), # cu1(pi/2) q[3],q[2];
    h(q[3]),                       # h q[3];
    measure(q, c)                  # measure q -> c;
)

program = current_program()
program = process(program, isolate_circuits=True)
sim = simulator.py(program)
sim.prepare()
while sim.run_step() :
    pass
sim.terminate()
