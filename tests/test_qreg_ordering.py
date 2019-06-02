from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
#from qgate.script.qelib1 import *
import numpy as np

class TestQregOrderingBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestQregOrderingBase:
            raise unittest.SkipTest()
        super(TestQregOrderingBase, cls).setUpClass()
    
    def test_empty_lanes(self) :
        qregs = new_qregs(3)
        circuit = []
        sim = self.create_simulator()
        sim.qubits.set_ordering(qregs)
        sim.run(circuit)
        prob = sim.qubits.prob[:]
        self.assertEqual(len(prob), 8)
        self.assertEqual(prob[0], 1.)
        self.assertTrue(np.all(prob[1:] == 0))
        
    def test_mixed_case(self) :
        qreg0, qreg1 = new_qregs(2)
        circuit = [I(qreg0)]
        sim = self.create_simulator()
        sim.qubits.set_ordering([qreg1])
        sim.run(circuit)
        prob = sim.qubits.prob[:]
        
        self.assertEqual(len(prob), 4)
        self.assertEqual(prob[0], 1.)
        self.assertTrue(np.all(prob[1:] == 0))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestQregOrdering', TestQregOrderingBase)
            
if __name__ == '__main__':
    unittest.main()
