from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.simulator.pyruntime import adjoint
from qgate.model.expand import expand
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
            product = np.matmul(mat, product)
        return product

    def test_expii(self) :
        qregs = new_qregs(4)
        circuit = [H(qreg) for qreg in qregs]
        states_h = self.run_sim(circuit)
        v = 1. / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_h, v))

        circuit += [ Expii(math.pi / 8)(qregs[0]) ]
        states_hexp = self.run_sim(circuit)

        v = cmath.exp(1.j * math.pi / 8) / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_hexp, v))

    def test_expiz(self) :
        qregs = new_qregs(4)
        circuit = [H(qreg) for qreg in qregs]
        states_h = self.run_sim(circuit)
        v = 1. / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_h, v))

        circuit += [ Expiz(math.pi / 8)(qregs[0]) ]
        states_hexp = self.run_sim(circuit)

        v = cmath.exp(1.j * math.pi / 8) / (math.sqrt(2) ** 4)
        self.assertTrue(np.allclose(states_hexp[0::2], v))
        self.assertTrue(np.allclose(states_hexp[1::2], np.conjugate(v)))
        
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestExp', TestExpBase)
                
if __name__ == '__main__':
    unittest.main()
