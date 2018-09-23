# OpenQASM Figure. 8
# // quantum teleportation example
#OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[3];
# creg c0[1];
# creg c1[1];
# creg c2[1];
# // optional post-rotation for state tomography
# gate post q { }
# u3(0.3,0.2,0.1) q[0];
# h q[1];
# cx q[1],q[2];
# barrier q;
# cx q[0],q[1];
# h q[0];
# measure q[0] -> c0[0];
# measure q[1] -> c1[0];
# if(c0==1) z q[2];
# if(c1==1) x q[2];
# post q[2];
# measure q[2] -> c2[0];



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


# qreg q[3];
q = allocate_qreg(3)
# creg c0[1];
c0 = allocate_creg(1)
# creg c1[1];
c1 = allocate_creg(1)
# creg c2[1];
c2 = allocate_creg(1)

op(
    # u3(0.3,0.2,0.1) q[0];
    u3(0.3,0.2,0.1, q[0]),
    # h q[1];
    h(q[1]),
    # cx q[1],q[2];
    cx(q[1], q[2]),
    # barrier q;
    barrier(q),
    # cx q[0],q[1];
    cx(q[0], q[1]),
    # h q[0];
    h(q[0]),
    # measure q[0] -> c0[0];
    measure(q[0], c0[0]),
    # measure q[1] -> c1[0];
    measure(q[1], c1[0]),
    # if(c0==1) z q[2];
    if_c(c0, 1, z(q[2])),
    # if(c1==1) x q[2];
    if_c(c1, 1, z(q[2])),

    # post q[2];
    #post(q[2]),
    # measure q[2] -> c2[0];
    #measure(q[2], c2[0])
    )

program = current_program()
process(program, seperate_circuits=True)
