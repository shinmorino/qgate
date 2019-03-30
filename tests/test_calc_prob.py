from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
#from qgate.script.qelib1 import *


class TestCalcProbBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestCalcProbBase:
            raise unittest.SkipTest()
        super(TestCalcProbBase, cls).setUpClass()

    def run_sim(self, circuit, isolate_circuits = False) :
        return self._run_sim(circuit, isolate_circuits)

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_calc_prob(self) :
        qreg = new_qreg()
        circuit = A(qreg)
        sim = self.run_sim(circuit)
        self.assertEqual(1., sim.qubits.calc_probability(qreg))
        
    def test_calc_prob_x(self) :
        qreg = new_qreg()
        circuit = X(qreg)
        sim = self.run_sim(circuit)
        self.assertAlmostEqual(0., sim.qubits.calc_probability(qreg))
        
    def test_calc_prob_multibits(self) :
        for qreg_idx in range(0, 10) :
            qregs = new_qregs(10)
            circuit = [
                [A(qreg) for qreg in qregs],
                A(qregs[qreg_idx])
            ]
            sim = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                self.assertAlmostEqual(1., sim.qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_multibits_x(self) :
        for qreg_idx in range(0, 10) :
            qregs = new_qregs(10)
            circuit = [
                [A(qreg) for qreg in qregs],
                X(qregs[qreg_idx])
            ]
            sim = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0., sim.qubits.calc_probability(qregs[obs_idx]))
                else :
                    self.assertAlmostEqual(1., sim.qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_multibits_h(self) :
        for qreg_idx in range(0, 10) :
            qregs = new_qregs(10)
            circuit = [
                [A(qreg) for qreg in qregs],
                H(qregs[qreg_idx])
            ]
            sim = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0.5, sim.qubits.calc_probability(qregs[obs_idx]))
                else :
                    self.assertAlmostEqual(1., sim.qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_cx_isolated(self) :
        qreg = new_qregs(10)
        circuit = [ X(qreg[0]),
                    X(qreg[2]),
                    X(qreg[4]),
                    X(qreg[6]),
                    X(qreg[8]),
                    ctrl(qreg[0]).X(qreg[1]),
                    ctrl(qreg[2]).X(qreg[3]),
                    ctrl(qreg[4]).X(qreg[5]),
                    ctrl(qreg[6]).X(qreg[7]),
                    ctrl(qreg[8]).X(qreg[9])
                ]
        sim = self.run_sim(circuit, True)
        for obs_idx in range(0, 10) :
            self.assertAlmostEqual(0., sim.qubits.calc_probability(qreg[obs_idx]))
                    
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestCalcProb', TestCalcProbBase)

if __name__ == '__main__':
    unittest.main()
