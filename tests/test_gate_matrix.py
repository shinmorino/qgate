from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.model.gate_type import U

import numpy as np
import math
import cmath

def gate_mat(gate) :
    # FIXME: Remove from test.
    mat = gate.gate_type.pymat()
    if gate.adjoint :
        return np.conjugate(mat.T)
    return mat

class TestGateMatrix(SimulatorTestBase) :

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)

    def assertAllClose(self, expected, actual) :
        unittest.TestCase.assertTrue(self, np.allclose(expected, actual, atol = 1.e-5))
        
    def test_id_gate(self) :
        qreg = new_qreg()
        id_ = I(qreg)
        self.assertAllClose([[1, 0], [0, 1]], gate_mat(id_))
        
    def test_x_gate(self) :
        qreg = new_qreg()
        x = X(qreg)
        self.assertAllClose([[0, 1], [1, 0]], gate_mat(x))

    def test_y_U_gate(self) :
        qreg = new_qreg()
        y = Y(qreg)
        self.assertAllClose([[0, -1.j], [1.j, 0]], gate_mat(y))
        
    def test_z_gate(self) :
        qreg = new_qreg()
        z = Z(qreg)
        self.assertAllClose([[1, 0], [0, -1]], gate_mat(z))

    def test_h_gate(self) :
        qreg = new_qreg()
        h = H(qreg)
        refmat = math.sqrt(0.5) * np.array([[1, 1], [1, -1]])
        self.assertAllClose(refmat, gate_mat(h))

    def test_S_gate(self) :
        qreg = new_qreg()
        s = S(qreg)
        refmat = np.array(([1, 0], [0, 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(s))

    def test_Sdg_gate(self) :
        qreg = new_qreg()
        sdg = S.Adj(qreg)
        refmat = np.array(([1, 0], [0, -1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(sdg))

    def test_T_gate(self) :
        qreg = new_qreg()
        t = T(qreg)
        refmat = np.array(([1, 0], [0, cmath.exp(1.j * math.pi / 4.)]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(t))

    def test_Tdg_gate(self) :
        qreg = new_qreg()
        tdg = T.Adj(qreg)
        refmat = np.array(([1, 0], [0, cmath.exp(-1.j * math.pi / 4.)]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(tdg))

    def test_Rx_gate(self) :
        qreg = new_qreg()
        
        rx = Rx(0)(qreg)
        refmat = np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rx))

        rx = Rx(math.pi * 2.)(qreg)
        refmat = np.array(([-1, 0], [0, -1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rx))

        rx = Rx(math.pi)(qreg)
        refmat = np.array(([0, -1.j], [-1.j, 0]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rx))

        rx = Rx(-math.pi)(qreg)
        refmat = np.array(([0, 1.j], [1.j, 0]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rx))
        
        rx = Rx(math.pi / 2.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1, -1.j], [-1.j, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rx))
        
        rx = Rx(- math.pi / 2.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1, 1.j], [1.j, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rx))
        

    def test_Ry_gate(self) :
        qreg = new_qreg()
        
        ry = Ry(0)(qreg)
        refmat = np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(ry))

        ry = Ry(math.pi * 2.)(qreg)
        refmat = np.array(([-1, 0], [0, -1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(ry))

        ry = Ry(math.pi)(qreg)
        refmat = np.array(([0, -1.], [1., 0]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(ry))

        ry = Ry(-math.pi)(qreg)
        refmat = np.array(([0, 1.], [-1., 0]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(ry))
        
        ry = Ry(math.pi / 2.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1, -1.], [1., 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(ry))
        
        ry = Ry(- math.pi / 2.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1, 1.], [-1., 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(ry))

    def test_Rz_gate(self) :
        qreg = new_qreg()
        
        rz = Rz(0)(qreg)
        refmat = np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rz))

        rz = Rz(math.pi * 2.)(qreg)
        refmat = np.array(([-1, 0], [0, -1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rz))

        rz = Rz(math.pi)(qreg)
        refmat = np.array(([-1.j, 0.], [0., 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rz))

        rz = Rz(-math.pi)(qreg)
        refmat = np.array(([1.j, 0.], [0., -1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rz))
        
        rz = Rz(math.pi / 2.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1. - 1.j, 0], [0., 1. + 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rz))
        
        rz = Rz(- math.pi / 2.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1. + 1.j, 0], [0., 1. - 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(rz))

    def test_ExpiI_gate(self) :
        qreg = new_qreg()
        
        expii = Expii(0)(qreg)
        refmat = np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expii))

        expii = Expii(math.pi)(qreg)
        refmat = -1. * np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expii))

        expii = Expii(math.pi / 2.)(qreg)
        refmat = 1.j * np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expii))

        expii = Expii(-math.pi / 2.)(qreg)
        refmat = -1.j * np.array(([1., 0.], [0., 1.]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expii))
        
        expii = Expii(math.pi / 4.)(qreg)
        refmat = math.sqrt(0.5) * (1. + 1.j) * np.array(([1., 0.], [0., 1.]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expii))
        
        expii = Expii(- math.pi / 4.)(qreg)
        refmat = math.sqrt(0.5) * (1. - 1.j) * np.array(([1., 0.], [0., 1.]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expii))

    def test_ExpiZ_gate(self) :
        qreg = new_qreg()
        
        expiz = Expiz(0)(qreg)
        refmat = np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expiz))

        expiz = Expiz(math.pi)(qreg)
        refmat = -1. * np.array(([1, 0], [0, 1]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expiz))

        expiz = Expiz(math.pi / 2.)(qreg)
        refmat = np.array(([1.j, 0], [0, -1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expiz))

        expiz = Expiz(-math.pi / 2.)(qreg)
        refmat = np.array(([-1.j, 0.], [0., 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expiz))
        
        expiz = Expiz(math.pi / 4.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1. + 1.j, 0.], [0., 1. - 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expiz))
        
        expiz = Expiz(- math.pi / 4.)(qreg)
        refmat = math.sqrt(0.5) * np.array(([1. - 1.j, 0.], [0., 1. + 1.j]), dtype=np.complex128)
        self.assertAllClose(refmat, gate_mat(expiz))


if __name__ == '__main__':
    unittest.main()
