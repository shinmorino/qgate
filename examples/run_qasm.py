import qgate.openqasm

qasm = '''
OPENQASM 2.0;

include "qelib1.inc";

qreg q[4];

h q[3];
cu1(pi/2) q[2], q[3];
cu1(pi/4) q[1], q[3];
cu1(pi/8) q[0], q[3];
h q[2];
cu1(pi/2) q[1], q[2];
cu1(pi/4) q[0], q[2];
h q[1];
cu1(pi/2) q[0], q[1];
h q[0];

creg c[4];
measure q -> c;
'''

code = qgate.openqasm.translate(qasm)
print(code)
mod = qgate.openqasm.load_circuit(qasm)

#print('q: ', mod.q)
#print('c: ', mod.c)
#
#print('circuit')
#qgate.dump(mod.circuit)

sim = qgate.simulator.cpu()
sim.run(mod.circuit)
print('qubit states')
qgate.dump(sim.qubits)
print('\nobservation: {}'.format(sim.obs(mod.c)))
