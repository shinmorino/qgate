from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

class TestResetBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestResetBase:
            raise unittest.SkipTest()
        super(TestResetBase, cls).setUpClass()
    
    def run_sim(self) :
        sim = self._run_sim()
        return sim.get_qubits().get_states(qgate.simulator.prob), sim.get_creg_dict()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_reset_0(self) :
        new_program()
        qreg = allocate_qreg(1)
        creg = allocate_creg(2)
        op(a(qreg), measure(qreg[0], creg[0]),
           reset(qreg[0]),
           measure(qreg[0], creg[1]))
        probs, creg_dict = self.run_sim()
        self.assertEqual(0, creg_dict.get_value(creg[0]))
        self.assertEqual(0, creg_dict.get_value(creg[1]))
        
    def test_reset_1(self) :
        new_program()
        qreg = allocate_qreg(1)
        creg = allocate_creg(2)
        op(x(qreg), measure(qreg[0], creg[0]),
           reset(qreg[0]),
           measure(qreg[0], creg[1]))
        probs, creg_dict = self.run_sim()
        self.assertEqual(1, creg_dict.get_value(creg[0]))
        self.assertEqual(0, creg_dict.get_value(creg[1]))
        
    def test_reset_not_allowed(self) :
        new_program()
        qreg = allocate_qreg(10)
        for idx in range(10) :
            op(reset(qreg[idx]))
            with self.assertRaises(RuntimeError) :
                self.run_sim()
        
    def test_reset_multi_qubits(self) :
        for idx in range(0, 10) :
            new_program()
            qreg = allocate_qreg(10)
            creg = allocate_creg(2)
            op(a(qreg[idx]), measure(qreg[idx], creg[0]),
               reset(qreg[idx]),
               measure(qreg[idx], creg[1]))
            probs, creg_dict = self.run_sim()
            self.assertEqual(0, creg_dict.get_value(creg[0]))
            self.assertEqual(0, creg_dict.get_value(creg[1]))

            new_program()
            qreg = allocate_qreg(10)
            creg = allocate_creg(2)
            op(x(qreg[idx]), measure(qreg[idx], creg[0]),
               reset(qreg[idx]),
               measure(qreg[idx], creg[1]))
            probs, creg_dict = self.run_sim()
            self.assertEqual(1, creg_dict.get_value(creg[0]))
            self.assertEqual(0, creg_dict.get_value(creg[1]))


import sys
this = sys.modules[__name__]
createTestCases(this, 'TestReset', TestResetBase)
            
if __name__ == '__main__':
    unittest.main()
