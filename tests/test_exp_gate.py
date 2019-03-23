from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.simulator.pyruntime import adjoint
from qgate.model.decompose import decompose
import math
import cmath
import numpy as np

class TestExpBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestExpBase:
            raise unittest.SkipTest()
        super(TestExpBase, cls).setUpClass()

    def run_sim(self, circuit) :
        sim = self._run_sim(circuit, isolate_circuits = False)
        states = sim.qubits.get_states()
        return states

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

    def test_expia(self) :
        qregs = new_qregs(4)
        circuit = new_circuit()
        circuit.add([H(qreg) for qreg in qregs])
        states_h = self.run_sim(circuit)
        v = 1. / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_h, v))

        circuit.add(Expia(math.pi / 8)(qregs[0]))
        states_hexp = self.run_sim(circuit)

        v = cmath.exp(1.j * math.pi / 8) / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_hexp, v))

    def test_expiz(self) :
        qregs = new_qregs(4)
        circuit = new_circuit()
        circuit.add([H(qreg) for qreg in qregs])
        states_h = self.run_sim(circuit)
        v = 1. / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_h, v))

        circuit.add(Expiz(math.pi / 8)(qregs[0]))
        states_hexp = self.run_sim(circuit)

        v = cmath.exp(1.j * math.pi / 8) / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_hexp[0::2], v))
        self.assertTrue(np.allclose(states_hexp[1::2], np.conjugate(v)))

    def test_exp_x(self) :
        theta = math.pi / 8.
        I = np.eye(2, dtype=np.complex128)
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128) 
        expix_ref = math.cos(theta) * I + 1.j * math.sin(theta) * x

        qreg = new_qreg()
        expix_dec = decompose(Expi(theta)(X(qreg)))
        expix_mat = self.gate_matrix_product(expix_dec)

        # print([gate.gate_type for gate in expix_dec])
        self.assertTrue(np.allclose(expix_ref, expix_mat))

    def test_exp_y(self) :
        theta = math.pi / 8.
        I = np.eye(2, dtype=np.complex128)
        y = np.array([[0, - 1.j], [1j, 0]], dtype=np.complex128) 
        expiy_ref = math.cos(theta) * I + 1.j * math.sin(theta) * y

        qreg = new_qreg()
        expiy_dec = decompose(Expi(theta)(Y(qreg)))
        expiy_mat = self.gate_matrix_product(expiy_dec)

        self.assertTrue(np.allclose(expiy_ref, expiy_mat))

    def test_exp_z(self) :
        theta = math.pi / 8.
        I = np.eye(2, dtype=np.complex128)
        z = np.array([[1, 0], [0, -1]], dtype=np.complex128) 
        expiz_ref = math.cos(theta) * I + 1.j * math.sin(theta) * z

        qreg = new_qreg()
        expiz_dec = decompose(Expi(theta)(Z(qreg)))
        expiz_mat = self.gate_matrix_product(expiz_dec)

        self.assertTrue(np.allclose(expiz_ref, expiz_mat))

    def test_exp_id(self) :
        theta = math.pi / 8.
        I = np.eye(2, dtype=np.complex128)
        expii_ref = cmath.exp(theta * 1.j) * I

        qreg = new_qreg()
        expii_dec = decompose(Expi(theta)(A(qreg)))
        expii_mat = self.gate_matrix_product(expii_dec)
        
        #print([gate.gate_type for gate in expii_dec])
        #print(expii_ref) print(expii_mat)
        self.assertTrue(np.allclose(expii_ref, expii_mat))

    def test_exp_xx(self) :
        theta = math.pi / 8.
        I = np.eye(2, dtype=np.complex128)
        expii_ref = cmath.exp(theta * 1.j) * I

        qreg = new_qreg()
        expii_dec = decompose(Expi(theta)(X(qreg), X(qreg)))
        expii_mat = self.gate_matrix_product(expii_dec)
        
        #print([gate.gate_type for gate in expii_dec])
        #print(expii_ref) print(expii_mat)
        self.assertTrue(np.allclose(expii_ref, expii_mat))
        
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestExp', TestExpBase)
                
if __name__ == '__main__':
    unittest.main()
