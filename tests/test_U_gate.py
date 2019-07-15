from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *

import numpy as np
import math
import cmath

def gate_mat(gate) :
    # FIXME: Remove from test.
    mat = gate.gate_type.pymat()
    if gate.adjoint :
        return np.conjugate(mat.T)
    return mat

class TestUGate(SimulatorTestBase) :

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)

    def assertAllClose(self, expected, actual) :
        unittest.TestCase.assertTrue(self, np.allclose(expected, actual, atol = 1.e-5))
    
    def test_id_U_gate(self) :
        refmat = [[1, 0], [0, 1]]
        qreg = new_qreg()

        u = U3(0, 0, 0)(qreg)
        self.assertAllClose(refmat, gate_mat(u))
        u = U1(0)(qreg)
        self.assertAllClose(refmat, gate_mat(u))
    
    def test_U2_gate(self) :
        qreg = new_qreg()

        u3 = U3(math.pi / 2., 0, 0)(qreg)
        u2 = U2(0, 0)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        # math.pi, 0.
        
        u3 = U3(math.pi / 2., math.pi, 0)(qreg)
        u2 = U2(math.pi, 0)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        u3 = U3(math.pi / 2., - math.pi, 0)(qreg)
        u2 = U2(- math.pi, 0)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        # math.pi / 2., 0.

        u3 = U3(math.pi / 2., math.pi / 2., 0)(qreg)
        u2 = U2(math.pi / 2., 0.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        u3 = U3(math.pi / 2., - math.pi / 2., 0)(qreg)
        u2 = U2(- math.pi / 2., 0.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        # math.pi / 4., 0.

        u3 = U3(math.pi / 2., math.pi / 4., 0)(qreg)
        u2 = U2(math.pi / 4., 0.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        u3 = U3(math.pi / 2., - math.pi / 4., 0)(qreg)
        u2 = U2(- math.pi / 4., 0.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        # 0., math.pi / 2.

        u3 = U3(math.pi / 2., 0., math.pi / 2.)(qreg)
        u2 = U2(0., math.pi / 2.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        u3 = U3(math.pi / 2., 0., - math.pi / 2.)(qreg)
        u2 = U2(0., - math.pi / 2.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        # 0., math.pi / 2.

        u3 = U3(math.pi / 2., 0., math.pi / 4.)(qreg)
        u2 = U2(0., math.pi / 4.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))

        u3 = U3(math.pi / 2., 0., - math.pi / 4.)(qreg)
        u2 = U2(0., - math.pi / 4.)(qreg)
        self.assertAllClose(gate_mat(u3), gate_mat(u2))
        
        
    def test_pauli_x_U_gate(self) :
        refmat = [[0, 1], [1, 0]]
        
        qreg = new_qreg()
        u = U3(math.pi, 0, math.pi)(qreg)
        self.assertAllClose(refmat, gate_mat(u) * 1.j)

    def test_pauli_y_U_gate(self) :
        refmat = [[0, - 1.j], [1.j, 0]]

        qreg = new_qreg()
        u = U3(math.pi, math.pi / 2., math.pi / 2.)(qreg)
        self.assertAllClose(refmat, gate_mat(u) * 1.j)
        
    def test_pauli_z_U_gate(self) :
        refmat = [[1, 0], [0, -1]]
        
        qreg = new_qreg()
        u = U3(0., 0., math.pi)(qreg)
        self.assertAllClose(refmat, gate_mat(u) * 1.j)
        
        u = U1(math.pi)(qreg)
        self.assertAllClose(refmat, gate_mat(u))

    def test_hadmard_U_gate(self) :
        refmat = math.sqrt(0.5) * np.array([[1, 1], [1, -1]])

        qreg = new_qreg()
        u = U3(math.pi / 2., 0., math.pi)(qreg)
        self.assertAllClose(refmat, gate_mat(u) * 1.j)
        
        u = U2(0., math.pi)(qreg)
        self.assertAllClose(refmat, gate_mat(u) * 1.j)

if __name__ == '__main__':
    unittest.main()
