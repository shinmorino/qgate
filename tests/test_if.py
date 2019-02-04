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
    
    def test_cx_gate_2qubits(self) :
        circuit = new_circuit()
        qregs = new_qregs(2)
        refs = new_references(2)
        circuit.add(measure(qregs[0], refs[0]),
                    if_(refs, 1, x(qregs[1])),
                    measure(qregs[1], refs[1]))
        sim = self.run_sim(circuit)
        self.assertEqual(0, sim.values.get(refs[0]))
        self.assertEqual(0, sim.values.get(refs[1]))
        
        circuit = new_circuit()
        qregs = new_qregs(2)
        refs = new_references(2)
        circuit.add(x(qregs[0]),
                    measure(qregs[0], refs[0]),
                    if_(refs, 1, x(qregs[1])),
                    measure(qregs[1], refs[1]))
        sim = self.run_sim(circuit)
        self.assertEqual(1, sim.values.get(refs[0]))
        self.assertEqual(1, sim.values.get(refs[1]))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestIf', TestIf)

if __name__ == '__main__':
    unittest.main()
