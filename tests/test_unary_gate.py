from __future__ import print_function
from __future__ import absolute_import

import qgate
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

import unittest

class TestUnaryGate(unittest.TestCase) :

    def run_sim(self) :
        program = current_program()
        program = qgate.qasm.process(program, isolate_circuits=False)
        sim = qgate.simulator.cpu(program)
        sim.prepare()
        sim.run()
        return sim.get_qubits().get_probabilities()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_id_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(a(qregs))
        probs = self.run_sim()
        self.assertEqual(probs[0], 1.)
        
    def test_pauli_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(x(qregs))
        probs = self.run_sim()
        self.assertEqual(probs[1], 1.)
        
    def test_pauli_gate_2(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(x(qregs), x(qregs))
        probs = self.run_sim()
        self.assertEqual(probs[0], 1.)
        
    def test_hadmard_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(h(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)
        
    def test_hadmard_gate2(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(h(qregs), h(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(probs[0], 1)
        self.assertAlmostEqual(probs[1], 0.)

    def test_pauli_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            new_program()
            qregs = allocate_qreg(n_qubits)
            op(x(qregs))
            probs = self.run_sim()
            self.assertAlmostEqual(probs[(1 << n_qubits) - 1], 1)

    def test_hadamard_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            new_program()
            qregs = allocate_qreg(n_qubits)
            op(h(qregs))
            probs = self.run_sim()
            n_states = 1 << n_qubits
            for idx in range(n_states) :
                self.assertAlmostEqual(probs[idx], 1. / n_states)


if __name__ == '__main__':
    unittest.main()
