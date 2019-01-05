from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.qelib1 import *
from qgate.qasm.script import *


class TestMeasureBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestMeasureBase:
            raise unittest.SkipTest()
        super(TestMeasureBase, cls).setUpClass()

    def run_sim(self) :
        sim = self._run_sim()
        return sim.get_qubits().get_states(qgate.simulator.prob), sim.get_creg_dict()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_measure_0(self) :
        new_program()
        qreg = allocate_qreg(1)
        creg = allocate_creg(1)
        op(a(qreg), measure(qreg[0], creg[0]))
        probs, creg_dict = self.run_sim()
        self.assertEqual(0, creg_dict.get_value(creg[0]))
        
    def test_measure_1(self) :
        new_program()
        qreg = allocate_qreg(1)
        creg = allocate_creg(1)
        op(x(qreg), measure(qreg[0], creg[0]))
        probs, creg_dict = self.run_sim()
        self.assertEqual(1, creg_dict.get_value(creg[0]))
        
    def test_measure_creg_array(self) :
        for qreg_idx in range(0, 10) :
            for creg_idx in range(0, 10) :
                new_program()
                qreg = allocate_qreg(10)
                creg = allocate_creg(10)
                op(a(qreg[qreg_idx]), measure(qreg[qreg_idx], creg[creg_idx]))
                probs, creg_dict = self.run_sim()
                self.assertEqual(0, creg_dict.get_value(creg[creg_idx]))

            new_program()
            qreg = allocate_qreg(10)
            creg = allocate_creg(10)
            op(x(qreg[qreg_idx]), measure(qreg[qreg_idx], creg[creg_idx]))
            probs, creg_dict = self.run_sim()
            self.assertEqual(1, creg_dict.get_value(creg[creg_idx]))
            self.assertEqual(1 << creg_idx, creg_dict.get_array_as_integer(creg))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestMeasure', TestMeasureBase)

if __name__ == '__main__':
    unittest.main()
