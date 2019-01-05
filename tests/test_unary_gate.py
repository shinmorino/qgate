from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

class TestUnaryGateBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestUnaryGateBase:
            raise unittest.SkipTest()
        super(TestUnaryGateBase, cls).setUpClass()
        
    def run_sim(self) :
        sim = self._run_sim()
        return sim.get_qubits().get_states(qgate.simulator.prob)
    
    def test_id_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(a(qregs))
        probs = self.run_sim()
        self.assertEqual(1., probs[0])
        
    def test_pauli_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(x(qregs))
        probs = self.run_sim()
        self.assertEqual(1., probs[1])
        
    def test_pauli_gate_2(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(x(qregs), x(qregs))
        probs = self.run_sim()
        self.assertEqual(1., probs[0])
        
    def test_hadmard_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(h(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(0.5, probs[0])
        self.assertAlmostEqual(0.5, probs[1])
        
    def test_hadmard_gate2(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(h(qregs), h(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(1., probs[0])
        self.assertAlmostEqual(0., probs[1])

    def test_pauli_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            new_program()
            qregs = allocate_qreg(n_qubits)
            op(x(qregs))
            probs = self.run_sim()
            self.assertAlmostEqual(1., probs[(1 << n_qubits) - 1])

    def test_pauli_gate_n_qubits(self) :
        n_qubits = 9
        new_program()
        qregs = allocate_qreg(n_qubits)
        op(x(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(1., probs[(1 << n_qubits) - 1])

    def test_hadmard_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            new_program()
            qregs = allocate_qreg(n_qubits)
            op(h(qregs))
            probs = self.run_sim()
            n_states = 1 << n_qubits
            for idx in range(n_states) :
                self.assertAlmostEqual(1. / n_states, probs[idx])

    def test_hadmard_gate_2_qubits(self) :
        n_qubits = 2
        new_program()
        qregs = allocate_qreg(n_qubits)
        op(h(qregs))
        probs = self.run_sim()
        n_states = 1 << n_qubits
        for idx in range(n_states) :
            self.assertAlmostEqual(1. / n_states, probs[idx])


import sys
this = sys.modules[__name__]
createTestCases(this, 'TestUnaryGate', TestUnaryGateBase)
                
if __name__ == '__main__':
    unittest.main()
