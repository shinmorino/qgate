import qgate
from qgate.script import *

qreg = new_qreg()
circuit = [ H(qreg), H(qreg) ]

pref = qgate.prefs.dynamic
sim = qgate.simulator.py(circuit_prep = pref)
sim.run(circuit)
sim.qubits.set_ordering([qreg])
probs = sim.qubits.prob[:]

print(probs)
