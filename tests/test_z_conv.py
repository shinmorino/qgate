from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.simulator.pyruntime import adjoint
import numpy as np

class TestZConvBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestZConvBase:
            raise unittest.SkipTest()
        super(TestZConvBase, cls).setUpClass()

    def get_matrix(self, gate) :
        mat = gate.gate_type.pymat()
        if gate.adjoint :
            mat = adjoint(mat)
        return mat

    def gate_matrix_product(self, ops) :
        product = np.eye(2, dtype=np.float64)
        for gate in ops :
            mat = self.get_matrix(gate)
            product = np.matmul(product, mat)
        return product

    def test_x_to_z(self) :
        qreg = new_qreg()
        d = h(qreg)
        _z = z(qreg)
        dadj = h(qreg)
        mat_hzh = self.gate_matrix_product([d, _z, dadj])

        _x = x(qreg)
        mat_x = self.gate_matrix_product([_x])
        self.assertTrue(np.allclose(mat_hzh, mat_x))
    
    def test_y_to_z(self) :
        qreg = new_qreg()
        d = sh(qreg)
        _z = z(qreg)
        dadj = sh.H(qreg)
        mat_sh_z_hsdg = self.gate_matrix_product([d, _z, dadj])

        _y = y(qreg)
        mat_y = self.gate_matrix_product([_y])
        self.assertTrue(np.allclose(mat_sh_z_hsdg, mat_y))
        
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestZConv', TestZConvBase)
                
if __name__ == '__main__':
    unittest.main()
