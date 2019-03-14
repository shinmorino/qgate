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
        circuit = new_circuit()
        qreg = new_qregs(1)
        circuit.add(a(qreg))
        sim = self.run_sim(circuit)
        self.assertEqual(1., sim.qubits.calc_probability(qreg[0]))
        
    def test_calc_prob_x(self) :
        circuit = new_circuit()
        qreg = new_qregs(1)
        circuit.add(x(qreg))
        sim = self.run_sim(circuit)
        self.assertAlmostEqual(0., sim.qubits.calc_probability(qreg[0]))
        
    def test_calc_prob_multibits(self) :
        for qreg_idx in range(0, 10) :
            circuit = new_circuit()
            qregs = new_qregs(10)
            circuit.add([a(qreg) for qreg in qregs],
                        a(qregs[qreg_idx]))
            sim = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                self.assertAlmostEqual(1., sim.qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_multibits_x(self) :
        for qreg_idx in range(0, 10) :
            circuit = new_circuit()
            qregs = new_qregs(10)
            circuit.add([a(qreg) for qreg in qregs],
                        x(qregs[qreg_idx]))
            sim = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0., sim.qubits.calc_probability(qregs[obs_idx]))
                else :
                    self.assertAlmostEqual(1., sim.qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_multibits_h(self) :
        for qreg_idx in range(0, 10) :
            circuit = new_circuit()
            qregs = new_qregs(10)
            circuit.add([a(qreg) for qreg in qregs],
                        h(qregs[qreg_idx]))
            sim = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0.5, sim.qubits.calc_probability(qregs[obs_idx]))
                else :
                    self.assertAlmostEqual(1., sim.qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_cx_isolated(self) :
        circuit = new_circuit()
        qreg = new_qregs(10)
        circuit.add(x(qreg[0]))
        circuit.add(x(qreg[2]))
        circuit.add(x(qreg[4]))
        circuit.add(x(qreg[6]))
        circuit.add(x(qreg[8]))
        circuit.add(ctrl(qreg[0]).x(qreg[1]))
        circuit.add(ctrl(qreg[2]).x(qreg[3]))
        circuit.add(ctrl(qreg[4]).x(qreg[5]))
        circuit.add(ctrl(qreg[6]).x(qreg[7]))
        circuit.add(ctrl(qreg[8]).x(qreg[9]))
        sim = self.run_sim(circuit, True)
        for obs_idx in range(0, 10) :
            self.assertAlmostEqual(0., sim.qubits.calc_probability(qreg[obs_idx]))
                    
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestCalcProb', TestCalcProbBase)

if __name__ == '__main__':
    unittest.main()
