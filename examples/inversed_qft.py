# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[4];
# creg c[4];
# h q;
# barrier q;
# h q[0];
# measure q[0] -> c[0];
# if(c==1) u1(pi/2) q[1];
# h q[1];
# measure q[1] -> c[1];
# if(c==1) u1(pi/4) q[2];
# if(c==2) u1(pi/2) q[2];
# if(c==3) u1(pi/2+pi/4) q[2];
# h q[2];
# measure q[2] -> c[2];
# if(c==1) u1(pi/8) q[3];
# if(c==2) u1(pi/4) q[3];
# if(c==3) u1(pi/4+pi/8) q[3];
# if(c==4) u1(pi/2) q[3];
# if(c==5) u1(pi/2+pi/8) q[3];
# if(c==6) u1(pi/2+pi/4) q[3];
# if(c==7) u1(pi/2+pi/4+pi/8) q[3];
# h q[3];
# measure q[3] -> c[3];


from qgate.script import *
import math

# include "qelib1.inc";
#from qgate.script.qelib1 import *

import qgate.simulator

circuit = new_circuit()


q = new_qregs(4) # qreg q[4];
c = new_references(4) # creg c[4];
circuit.add (
    [h(_q) for _q in q],                                # h q;
    barrier(q),                                         # barrier q;
    h(q[0]),                                            # h q[0];
    measure(c[0], q[0]),                                # measure q[0] -> c[0];
    if_(c, 1, u1(math.pi / 2.)(q[1])),                  # if(c==1) u1(pi/2) q[1];
    h(q[1]),                                            # h q[1];
    measure(c[1], q[1]),                                # measure q[1] -> c[1];
    if_(c, 1, u1(math.pi / 4.)(q[2])),                  # if(c==1) u1(pi/4) q[2];
    if_(c, 2, u1(math.pi / 2.)(q[2])),                  # if(c==2) u1(pi/2) q[2];
    if_(c, 3, u1(math.pi / 2. + math.pi / 4.)(q[2])),   # if(c==3) u1(pi/2+pi/4) q[2];
    h(q[2]),                                            # h q[2];
    measure(c[2], q[2]),                                # measure q[2] -> c[2];
    if_(c, 1, u1(math.pi / 8.)(q[3])),                  # if(c==1) u1(pi/8) q[3];
    if_(c, 2, u1(math.pi / 4.)(q[3])),                  # if(c==2) u1(pi/4) q[3];
    if_(c, 3, u1(math.pi / 4. + math.pi / 8.)(q[3])),   # if(c==3) u1(pi/4+pi/8) q[3];
    if_(c, 4, u1(math.pi / 2.)(q[3])),                  # if(c==4) u1(pi/2) q[3];
    if_(c, 5, u1(math.pi / 2. + math.pi / 8.)(q[3])),   # if(c==5) u1(pi/2+pi/8) q[3];
    if_(c, 6, u1(math.pi / 2. + math.pi / 4.)(q[3])),   # if(c==6) u1(pi/2+pi/4) q[3];
    if_(c, 7, u1(math.pi / 2. + math.pi / 4. + math.pi / 8.)(q[3])),
                                                        # if(c==7) u1(pi/2+pi/4+pi/8) q[3];
    h(q[3]),                                            # h q[3];
    measure(c[3], q[3])                                 # measure q[3] -> c[3];
)

sim = qgate.simulator.py(isolate_circuits=True)
sim.run(circuit)
sim.terminate()
