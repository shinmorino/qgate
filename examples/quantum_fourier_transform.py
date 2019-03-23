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


import qgate
from qgate.script import *
import math

# from qgate.script.qelib1 import *  # include "qelib1.inc";

circuit = new_circuit()

# // optional post-rotation for state tomography
# gate post q { }
def post(qregs) :
    return a(qregs)



q = new_qregs(4) # qreg q[4];
c = new_references(4) # creg c[4];
circuit.add(
    X(q[0]),                       # x q[0]
    X(q[2]),                       # x q[2];
    barrier(q),                    # barrier q;
    H(q[0]),                       # h q[0];
    ctrl(q[1]).U1(math.pi / 2.)(q[0]), # cu1(pi/2) q[1],q[0];
    H(q[1]),                       # h q[1];
    ctrl(q[2]).U1(math.pi / 4.)(q[0]), # cu1(pi/4) q[2],q[0];
    ctrl(q[2]).U1(math.pi / 2.)(q[1]), # cu1(pi/2) q[2],q[1];
    H(q[2]),                       # h q[2];
    ctrl(q[3]).U1(math.pi / 8.)(q[0]), # cu1(pi/8) q[3],q[0];
    ctrl(q[3]).U1(math.pi / 4.)(q[1]), # cu1(pi/4) q[3],q[1];
    ctrl(q[3]).U1(math.pi / 2.)(q[2]), # cu1(pi/2) q[3],q[2];
    H(q[3]),                       # h q[3];
    [measure(_c, _q) for _c, _q in zip (c, q)] # measure q -> c;
)

sim = qgate.simulator.py(isolate_circuits=True)
sim.run(circuit)
sim.terminate()
