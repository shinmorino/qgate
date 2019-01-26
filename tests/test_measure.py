from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.script.qelib1 import *


class TestMeasureBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestMeasureBase:
            raise unittest.SkipTest()
        super(TestMeasureBase, cls).setUpClass()

    def run_sim(self, circuit) :
        sim = self._run_sim(circuit)
        return sim.qubits().get_states(qgate.simulator.prob), sim.creg_values()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_measure_0(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        cregs = allocate_cregs(1)
        circuit.add(a(qregs), measure(qregs[0], cregs[0]))
        probs, creg_values = self.run_sim(circuit)
        self.assertEqual(0, creg_values.get(cregs[0]))
        
    def test_measure_1(self) :
        circuit = new_circuit()
        qregs = allocate_qregs(1)
        cregs = allocate_cregs(1)
        circuit.add(x(qregs), measure(qregs[0], cregs[0]))
        probs, creg_values = self.run_sim(circuit)
        self.assertEqual(1, creg_values.get(cregs[0]))
        
    def test_measure_cregs_array(self) :
        for qregs_idx in range(0, 10) :
            for cregs_idx in range(0, 10) :
                circuit = new_circuit()
                qregs = allocate_qregs(10)
                cregs = allocate_cregs(10)
                circuit.add(a(qregs[qregs_idx]), measure(qregs[qregs_idx], cregs[cregs_idx]))
                probs, creg_values = self.run_sim(circuit)
                self.assertEqual(0, creg_values.get(cregs[cregs_idx]))

            circuit = new_circuit()
            qregs = allocate_qregs(10)
            cregs = allocate_cregs(10)
            circuit.add(x(qregs[qregs_idx]), measure(qregs[qregs_idx], cregs[cregs_idx]))
            probs, creg_values = self.run_sim(circuit)
            self.assertEqual(1, creg_values.get(cregs[cregs_idx]))
            self.assertEqual(1 << cregs_idx, creg_values.get_packed_value(cregs))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestMeasure', TestMeasureBase)

if __name__ == '__main__':
    unittest.main()
