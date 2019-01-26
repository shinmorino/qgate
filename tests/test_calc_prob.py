from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.script.qelib1 import *


class TestCalcProbBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestCalcProbBase:
            raise unittest.SkipTest()
        super(TestCalcProbBase, cls).setUpClass()

    def run_sim(self, circuit, isolate_circuits = False) :
        sim = self._run_sim(circuit, isolate_circuits)
        return sim.qubits()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_calc_prob(self) :
        circuit = new_circuit()
        qreg = allocate_qregs(1)
        creg = allocate_cregs(1)
        circuit.add(a(qreg))
        qubits = self.run_sim(circuit)
        self.assertEqual(1., qubits.calc_probability(qreg[0]))
        
    def test_calc_prob_x(self) :
        circuit = new_circuit()
        qreg = allocate_qregs(1)
        creg = allocate_cregs(1)
        circuit.add(x(qreg))
        qubits = self.run_sim(circuit)
        self.assertAlmostEqual(0., qubits.calc_probability(qreg[0]))
        
    def test_calc_prob_multibits(self) :
        for qreg_idx in range(0, 10) :
            circuit = new_circuit()
            qregs = allocate_qregs(10)
            circuit.add(a(qregs),
                        a(qregs[qreg_idx]))
            qubits = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                self.assertAlmostEqual(1., qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_multibits_x(self) :
        for qreg_idx in range(0, 10) :
            circuit = new_circuit()
            qregs = allocate_qregs(10)
            circuit.add(a(qregs),
                        x(qregs[qreg_idx]))
            qubits = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0., qubits.calc_probability(qregs[obs_idx]))
                else :
                    self.assertAlmostEqual(1., qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_multibits_h(self) :
        for qreg_idx in range(0, 10) :
            circuit = new_circuit()
            qregs = allocate_qregs(10)
            circuit.add(a(qregs),
                        h(qregs[qreg_idx]))
            qubits = self.run_sim(circuit)
            for obs_idx in range(0, 10) :
                if qreg_idx == obs_idx :
                    self.assertAlmostEqual(0.5, qubits.calc_probability(qregs[obs_idx]))
                else :
                    self.assertAlmostEqual(1., qubits.calc_probability(qregs[obs_idx]))

    def test_calc_prob_cx_isolated(self) :
        circuit = new_circuit()
        qreg = allocate_qregs(10)
        creg = allocate_cregs(10)
        circuit.add(x(qreg[0]))
        circuit.add(x(qreg[2]))
        circuit.add(x(qreg[4]))
        circuit.add(x(qreg[6]))
        circuit.add(x(qreg[8]))
        circuit.add(cx(qreg[0], qreg[1]))
        circuit.add(cx(qreg[2], qreg[3]))
        circuit.add(cx(qreg[4], qreg[5]))
        circuit.add(cx(qreg[6], qreg[7]))
        circuit.add(cx(qreg[8], qreg[9]))
        qubits = self.run_sim(circuit, True)
        for obs_idx in range(0, 10) :
            self.assertAlmostEqual(0., qubits.calc_probability(qreg[obs_idx]))
                    
import sys
this = sys.modules[__name__]
createTestCases(this, 'TestCalcProb', TestCalcProbBase)

if __name__ == '__main__':
    unittest.main()