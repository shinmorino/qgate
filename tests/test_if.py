from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
#from qgate.script.qelib1 import *

class TestIf(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestIf:
            raise unittest.SkipTest()
        super(TestIf, cls).setUpClass()
        
    def run_sim(self, circuit) :
        return self._run_sim(circuit)

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)

    def test_x(self) :
        qregs = new_qregs(2)
        ref = new_reference()
        circuit = [
            X(qregs[0]),
            measure(ref, qregs[0]),
            if_(ref, 1, X(qregs[1])),
        ]
        sim = self.run_sim(circuit)
        self.assertEqual(1, sim.qubits.prob[3])
    
    def test_cx_gate_2qubits(self) :
        qregs = new_qregs(2)
        refs = new_references(2)
        circuit = [
            measure(refs[0], qregs[0]),
            if_(refs, 1, X(qregs[1])),
            measure(refs[1], qregs[1])
        ]
        sim = self.run_sim(circuit)
        self.assertEqual(0, sim.values.get(refs[0]))
        self.assertEqual(0, sim.values.get(refs[1]))
        
        qregs = new_qregs(2)
        refs = new_references(2)
        circuit = [
            X(qregs[0]),
            measure(refs[0], qregs[0]),
            if_(refs, 1, X(qregs[1])),
            measure(refs[1], qregs[1])
        ]
        sim = self.run_sim(circuit)
        self.assertEqual(1, sim.values.get(refs[0]))
        self.assertEqual(1, sim.values.get(refs[1]))
    
    def test_if_pred(self) :
        qregs = new_qregs(2)
        refs = new_references(2)
        circuit = [
            measure(refs[0], qregs[0]),
            if_(refs,
                lambda v0, v1: v0 == 1 and v1 == None,
                X(qregs[1])),
            measure(refs[1], qregs[1])
        ]
        sim = self.run_sim(circuit)
        self.assertEqual(0, sim.values.get(refs[0]))
        self.assertEqual(0, sim.values.get(refs[1]))
        
        qregs = new_qregs(2)
        refs = new_references(2)
        circuit = [
            X(qregs[0]),
            measure(refs[0], qregs[0]),
            if_(refs,
                lambda v0, v1: v0 == 1 and v1 == None,
                X(qregs[1])),
            measure(refs[1], qregs[1])
        ]
        sim = self.run_sim(circuit)
        self.assertEqual(1, sim.values.get(refs[0]))
        self.assertEqual(1, sim.values.get(refs[1]))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestIf', TestIf)

if __name__ == '__main__':
    unittest.main()
