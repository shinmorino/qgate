from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.qelib1 import *
from qgate.qasm.script import *


class TestCalcProbBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestCalcProbBase:
            raise unittest.SkipTest()
        super(TestCalcProbBase, cls).setUpClass()

    def run_sim(self) :
        sim = self._run_sim()
        return sim.get_qubits()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_calc_prob(self) :
        new_program()
        qreg = allocate_qreg(1)
        creg = allocate_creg(1)
        op(a(qreg))
        qubits = self.run_sim()
        self.assertEqual(1., qubits.calc_probability(qreg[0]))
        
    def test_calc_prob_x(self) :
        new_program()
        qreg = allocate_qreg(1)
        creg = allocate_creg(1)
        op(x(qreg))
        qubits = self.run_sim()
        self.assertAlmostEqual(0., qubits.calc_probability(qreg[0]))
        
    def test_calc_prob_multibits(self) :
        for qreg_idx in range(0, 10) :
            new_program()
            qreg = allocate_qreg(10)
            creg = allocate_creg(10)
            op(a(qreg[qreg_idx]))
            qubits = self.run_sim()
            for obs_idx in range(0, 10) :
                self.assertAlmostEqual(1., qubits.calc_probability(qreg[obs_idx]))

    def test_calc_prob_multibits_x(self) :
        for qreg_idx in range(0, 10) :
            new_program()
            qreg = allocate_qreg(10)
            creg = allocate_creg(10)
            op(x(qreg[qreg_idx]))
            qubits = self.run_sim()
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0., qubits.calc_probability(qreg[obs_idx]))
                else :
                    self.assertAlmostEqual(1., qubits.calc_probability(qreg[obs_idx]))

    def test_calc_prob_multibits_h(self) :
        for qreg_idx in range(0, 10) :
            new_program()
            qreg = allocate_qreg(10)
            creg = allocate_creg(10)
            op(h(qreg[qreg_idx]))
            qubits = self.run_sim()
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0.5, qubits.calc_probability(qreg[obs_idx]))
                else :
                    self.assertAlmostEqual(1., qubits.calc_probability(qreg[obs_idx]))
                    
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestCalcProb', TestCalcProbBase)

if __name__ == '__main__':
    unittest.main()
