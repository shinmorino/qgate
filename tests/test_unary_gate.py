from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.script.qelib1 import *

class TestUnaryGateBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestUnaryGateBase:
            raise unittest.SkipTest()
        super(TestUnaryGateBase, cls).setUpClass()
        
    def run_sim(self, circuit) :
        sim = self._run_sim(circuit)
        return sim.qubits().get_states(qgate.simulator.prob)
    
    def test_id_gate(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        circuit.add(a(qregs))
        probs = self.run_sim(circuit)
        self.assertEqual(1., probs[0])
        
    def test_pauli_gate(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        circuit.add(x(qregs))
        probs = self.run_sim(circuit)
        self.assertEqual(1., probs[1])
        
    def test_pauli_gate_2(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        circuit.add(x(qregs), x(qregs))
        probs = self.run_sim(circuit)
        self.assertEqual(1., probs[0])
        
    def test_hadmard_gate(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        circuit.add(h(qregs))
        probs = self.run_sim(circuit)
        self.assertAlmostEqual(0.5, probs[0])
        self.assertAlmostEqual(0.5, probs[1])
        
    def test_hadmard_gate2(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        circuit.add(h(qregs), h(qregs))
        probs = self.run_sim(circuit)
        self.assertAlmostEqual(1., probs[0])
        self.assertAlmostEqual(0., probs[1])

    def test_pauli_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            circuit = new_circuit()
            qregs = allocate_qregs(n_qubits)
            circuit.add(x(qregs))
            probs = self.run_sim(circuit)
            self.assertAlmostEqual(1., probs[(1 << n_qubits) - 1])

    def test_pauli_gate_n_qubits(self) :
        n_qubits = 9
        circuit = new_circuit()
        qregs = allocate_qregs(n_qubits)
        circuit.add(x(qregs))
        probs = self.run_sim(circuit)
        self.assertAlmostEqual(1., probs[(1 << n_qubits) - 1])

    def test_hadmard_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            circuit = new_circuit()
            qregs = allocate_qregs(n_qubits)
            circuit.add(h(qregs))
            probs = self.run_sim(circuit)
            n_states = 1 << n_qubits
            for idx in range(n_states) :
                self.assertAlmostEqual(1. / n_states, probs[idx])

    def test_hadmard_gate_2_qubits(self) :
        n_qubits = 2
        circuit = new_circuit()
        qregs = allocate_qregs(n_qubits)
        circuit.add(h(qregs))
        probs = self.run_sim(circuit)
        n_states = 1 << n_qubits
        for idx in range(n_states) :
            self.assertAlmostEqual(1. / n_states, probs[idx])


import sys
this = sys.modules[__name__]
createTestCases(this, 'TestUnaryGate', TestUnaryGateBase)
                
if __name__ == '__main__':
    unittest.main()
