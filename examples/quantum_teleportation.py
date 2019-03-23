# OpenQASM Figure. 8
# // quantum teleportation example
# OPENQASM 2.0;
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



from qgate.script import *
# include "qelib1.inc";
# from qgate.script.qelib1 import *

import qgate.simulator.simulator

# // optional post-rotation for state tomography
# gate post q { }
def post(qregs) :
    return A(qregs)


# qreg q[3];
q = new_qregs(3)
# creg c0[1];
c0 = new_reference()
# creg c1[1];
c1 = new_reference()
# creg c2[1];
c2 = new_reference()

circuit = new_circuit()

circuit.add(
    U3(0.3,0.2,0.1) (q[0]), # u3(0.3,0.2,0.1) q[0];
    H(q[1]),                # h q[1];
    ctrl(q[1]).X(q[2]),     # cx q[1],q[2];
    barrier(q),             # barrier q;
    ctrl(q[0]).X(q[1]),     # cx q[0],q[1];
    H(q[0]),                # h q[0];
    measure(c0, q[0]),      # measure q[0] -> c0[0];
    measure(c1, q[1]),      # measure q[1] -> c1[0];
    if_(c0, 1)(Z(q[2])),    # if(c0==1) z q[2];
    if_(c1, 1)(X(q[2])),    # if(c1==1) x q[2];

    post(q[2]),             # post q[2];
    measure(c2, q[2])       # measure q[2] -> c2[0];
)

sim = qgate.simulator.py()
sim.run(circuit)

qgate.dump(sim.qubits)
qgate.dump(sim.values)

sim.terminate()
