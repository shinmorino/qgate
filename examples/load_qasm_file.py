import qgate.openqasm

# example to load and run files.

filepath = 'qft.qasm'

# generate source module
qasm = qgate.openqasm.translate_file(filepath)
print(qasm)
    
mod = qgate.openqasm.load_circuit_from_file(filepath)
sim = qgate.simulator.cpu()
sim.run(mod.circuit)

qgate.dump(sim.qubits)
qgate.dump(sim.qubits.prob)

