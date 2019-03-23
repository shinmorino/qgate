from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.model.gate_type import U

import numpy as np
import math

class TestUGateBase(SimulatorTestBase) :

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)

    def assertAllClose(self, expected, actual) :
        unittest.TestCase.assertTrue(self, np.allclose(expected, actual, atol = 1.e-5))
    
    def test_id_U_gate(self) :
        u = U(0, 0, 0)
        mat = u.pymat()
        self.assertAllClose([[1, 0], [0, 1]], mat)
        
    def test_pauli_x_U_gate(self) :
        u = U(math.pi, 0, math.pi)
        mat = u.pymat()
        self.assertAllClose([[0, 1], [1, 0]], mat)

        u = U3(math.pi, 0, math.pi)(new_qreg())
        mat = u.gate_type.pymat()
        self.assertAllClose([[0, 1], [1, 0]], mat)

    def test_pauli_y_U_gate(self) :
        u = U(math.pi, math.pi / 2., math.pi / 2.)
        mat = u.pymat()
        self.assertAllClose(mat, [[0, - 1.j], [1.j, 0]])
        
        u = U3(math.pi, math.pi / 2., math.pi / 2.)(new_qreg())
        mat = u.gate_type.pymat()
        self.assertAllClose([[0, - 1.j], [1.j, 0]], mat)
        
    def test_pauli_z_U_gate(self) :
        u = U(0., 0., math.pi)
        mat = u.pymat()
        self.assertAllClose([[1, 0], [0, -1]], mat)
        
        u = U1(math.pi)(new_qreg())
        mat = u.gate_type.pymat()
        self.assertAllClose([[1, 0], [0, -1]], mat)

    def test_hadmard_U_gate(self) :
        h = math.sqrt(0.5) * np.array([[1, 1], [1, -1]])
        u = U(math.pi / 2., 0., math.pi)
        mat = u.pymat()
        self.assertAllClose(h, mat)
        
        u = U2(0., math.pi)(new_qreg())
        mat = u.gate_type.pymat()
        self.assertAllClose(h, mat)


if __name__ == '__main__':
    unittest.main()
