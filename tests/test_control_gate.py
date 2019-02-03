from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *

class TestControlGateBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestControlGateBase:
            raise unittest.SkipTest()
        super(TestControlGateBase, cls).setUpClass()

    def run_sim(self, circuit) :
        qubits = self._run_sim(circuit).qubits()
        probs = qubits.get_states(qgate.simulator.prob)
        return qubits, probs
    
    def test_cx_gate_2qubits(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(2)                # |00>
        circuit.add(cntr(qregs[0]).x(qregs[1]))  # |00>
        qubits, probs = self.run_sim(circuit)
        self.assertAlmostEqual(1, probs[0])

        circuit = new_circuit()
        qregs = allocate_qregs(2)
        circuit.add(x(qregs[0]))                 # |01>
        circuit.add(cntr(qregs[0]).x(qregs[1]))  # |11>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.get_state_index(*qregs)
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = new_circuit()
        qregs = allocate_qregs(2)
        circuit.add(x(qregs[1]))                 # |10>
        circuit.add(cntr(qregs[0]).x(qregs[1]))  # |10>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.get_state_index(qregs[1])
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = new_circuit()
        qregs = allocate_qregs(2)
        circuit.add(x(qregs))                    # |11>
        circuit.add(cntr(qregs[0]).x(qregs[1]))  # |01>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.get_state_index(qregs[0])
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = new_circuit()
        qregs = allocate_qregs(2)                # |00>
        circuit.add(cntr(qregs[1]).x(qregs[0]))  # |00>
        qubits, probs = self.run_sim(circuit)
        self.assertAlmostEqual(1, probs[0])

        circuit = new_circuit()
        qregs = allocate_qregs(2)
        circuit.add(x(qregs[0]))                 # |01>
        circuit.add(cntr(qregs[1]).x(qregs[0]))  # |01>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.get_state_index(qregs[0])
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = new_circuit()
        qregs = allocate_qregs(2)
        circuit.add(x(qregs[1]))                 # |10>
        circuit.add(cntr(qregs[1]).x(qregs[0]))  # |11>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.get_state_index(*qregs)
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = new_circuit()
        qregs = allocate_qregs(2)
        circuit.add(x(qregs))                    # |11>
        circuit.add(cntr(qregs[1]).x(qregs[0]))  # |10>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.get_state_index(qregs[1])
        self.assertAlmostEqual(1, probs[state_idx])

    def test_cx_gate_multibits(self) :

        for n_qregs in range(1, 9) :
            for control in range(0, n_qregs) :
                for target in range(0, n_qregs) :
                    if control == target :
                        continue
                    
                    circuit = new_circuit()
                    qregs = allocate_qregs(n_qregs)
                    circuit.add(a(qregs))
                    circuit.add(cntr(qregs[control]).x(qregs[target]))
                    qubits, probs = self.run_sim(circuit)
                    self.assertAlmostEqual(1, probs[0])
                    
                    circuit = new_circuit()
                    qregs = allocate_qregs(n_qregs)
                    circuit.add(a(qregs))
                    circuit.add(x(qregs[control]))
                    circuit.add(cntr(qregs[control]).x(qregs[target]))
                    qubits, probs = self.run_sim(circuit)
                    ext_idx = qubits.get_state_index(qregs[control], qregs[target])
                    
                    self.assertAlmostEqual(1, probs[ext_idx])

    def test_cx_gate_multibits_2(self) :

        n_qregs = 3
        control = 2
        target = 0

        circuit = new_circuit()
        qregs = allocate_qregs(n_qregs)
        circuit.add(a(qregs))
        circuit.add(x(qregs[control]))
        circuit.add(cntr(qregs[control]).x(qregs[target]))
        qubits = self._run_sim(circuit).qubits()
        probs = qubits.get_states(qgate.simulator.prob)
        ext_idx = qubits.get_state_index(qregs[control], qregs[target])
        self.assertAlmostEqual(1, probs[ext_idx])
                

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestControlGate', TestControlGateBase)

if __name__ == '__main__':
    unittest.main()
