from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

class TestIf(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestIf:
            raise unittest.SkipTest()
        super(TestIf, cls).setUpClass()
        
    def run_sim(self) :
        sim = self._run_sim()
        return sim.get_qubits(), sim.get_creg_dict()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_cx_gate_2qubits(self) :
        new_program()
        qregs = allocate_qreg(2)
        cregs = allocate_creg(2)
        op(measure(qregs[0], cregs[0]),
           if_(cregs, 1, x(qregs[1])),
           measure(qregs[1], cregs[1]))
        qubits, creg_dict = self.run_sim()
        self.assertEqual(0, creg_dict.get_value(cregs[0]))
        self.assertEqual(0, creg_dict.get_value(cregs[1]))
        
        new_program()
        qregs = allocate_qreg(2)
        cregs = allocate_creg(2)
        op(x(qregs[0]),
           measure(qregs[0], cregs[0]),
           if_(cregs, 1, x(qregs[1])),
           measure(qregs[1], cregs[1]))
        qubits, creg_dict = self.run_sim()
        self.assertEqual(1, creg_dict.get_value(cregs[0]))
        self.assertEqual(1, creg_dict.get_value(cregs[1]))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestIf', TestIf)

if __name__ == '__main__':
    unittest.main()
