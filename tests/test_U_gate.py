from __future__ import print_function
from __future__ import absolute_import

import qgate
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *
import math

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

    def assertAllClose(self, expected, actual) :
        unittest.TestCase.assertTrue(self, np.allclose(expected, actual, atol = 1.e-5))
    
    def test_id_U_gate(self) :
        u = U(0, 0, 0, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[1, 0], [0, 1]])
        
    def test_pauli_x_U_gate(self) :
        u = U(math.pi, 0, math.pi, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[0, 1], [1, 0]])

        u = u3(math.pi, 0, math.pi, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[0, 1], [1, 0]])

    def test_pauli_y_U_gate(self) :
        u = U(math.pi, math.pi / 2., math.pi / 2., None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[0, - 1.j], [1.j, 0]])
        
        u = u3(math.pi, math.pi / 2., math.pi / 2., None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[0, - 1.j], [1.j, 0]])
        
    def test_pauli_z_U_gate(self) :
        u = U(0., 0., math.pi, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[1, 0], [0, -1]])
        
        u = u1(math.pi, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, [[1, 0], [0, -1]])

    def test_hadmard_U_gate(self) :
        h = math.sqrt(0.5) * np.array([[1, 1], [1, -1]])
        u = U(math.pi / 2., 0., math.pi, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, h)
        
        u = u2(0., math.pi, None)
        mat = u.get_matrix()
        self.assertAllClose(mat, h)


if __name__ == '__main__':
    unittest.main()
