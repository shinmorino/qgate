from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append('C:\\projects\\qgate_sandbox')

from tests.test_base import *
from qgate.script import *

class TestBigCircuitsBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestBigCircuitsBase:
            raise unittest.SkipTest()
        super(TestBigCircuitsBase, cls).setUpClass()
    
    def setUp(self) :
        self.n_qregs = 20
        
    def run_sim(self, circuit) :
        return self._run_sim(circuit, False)

    def test_hadmard_gate(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit)
        for lane in range(0, self.n_qregs) :
            prob = sim.qubits.calc_probability(qregs[lane])
            # print('lane {}'.format(lane))
            self.assertAlmostEqual(0.5, prob)

    def test_x_gate(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([x(qreg) for qreg in qregs])
        sim = self.run_sim(circuit)
        for lane in range(0, self.n_qregs) :
            prob = sim.qubits.calc_probability(qregs[lane])
            # print('lane {}'.format(lane))
            self.assertAlmostEqual(0., prob)

    def test_cx_gate(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add(x(qregs[0]))
        for idx in range(0, self.n_qregs - 1) :
            circuit.add(ctrl(qregs[idx]).x(qregs[idx + 1]))
        sim = self.run_sim(circuit)
        for lane in range(0, self.n_qregs) :
            prob = sim.qubits.calc_probability(qregs[lane])
            # print('lane {}'.format(lane))
            self.assertAlmostEqual(0., prob)
            
import sys
this = sys.modules[__name__]

createCPUTestCase(this, 'TestBigCircuits', TestBigCircuitsBase)
createCUDATestCase(this, 'TestBigCircuits', TestBigCircuitsBase)

if __name__ == '__main__':
    unittest.main()
