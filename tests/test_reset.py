from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
#from qgate.script.qelib1 import *

class TestResetBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestResetBase:
            raise unittest.SkipTest()
        super(TestResetBase, cls).setUpClass()
    
    def run_sim(self, circuit) :
        return self._run_sim(circuit)

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_reset_0(self) :
        circuit = new_circuit()
        qreg = new_qreg()
        refs = new_references(2)
        circuit.add(a(qreg),
                    measure(refs[0], qreg),
                    reset(qreg),
                    measure(refs[1], qreg)
                    # pmeasure(refs[1])(z(qreg[0]))
        )
        sim = self.run_sim(circuit)
        self.assertEqual(0, sim.values.get(refs[0]))
        self.assertEqual(0, sim.values.get(refs[1]))
        
    def test_reset_1(self) :
        circuit = new_circuit()
        qreg = new_qreg()
        refs = new_references(2)
        circuit.add(x(qreg),
                    measure(refs[0], qreg),
                    reset(qreg),
                    measure(refs[1], qreg))
        sim = self.run_sim(circuit)
        self.assertEqual(1, sim.values.get(refs[0]))
        self.assertEqual(0, sim.values.get(refs[1]))
        
    def test_reset_not_allowed(self) :
        circuit = new_circuit()
        qreg = new_qregs(10)
        for idx in range(10) :
            circuit.add(reset(qreg[idx]))
            with self.assertRaises(RuntimeError) :
                self.run_sim(circuit)
        
    def test_reset_multi_qubits(self) :
        for idx in range(0, 10) :
            circuit = new_circuit()
            qreg = new_qregs(10)
            refs = new_references(2)
            circuit.add(a(qreg[idx]), measure(refs[0], qreg[idx]),
                        reset(qreg[idx]),
                        measure(refs[1], qreg[idx]))
            sim = self.run_sim(circuit)
            self.assertEqual(0, sim.values.get(refs[0]))
            self.assertEqual(0, sim.values.get(refs[1]))

            circuit = new_circuit()
            qreg = new_qregs(10)
            refs = new_references(2)
            circuit.add(x(qreg[idx]),
                        measure(refs[0], qreg[idx]),
                        reset(qreg[idx]),
                        measure(refs[1], qreg[idx]))
            sim = self.run_sim(circuit)
            self.assertEqual(1, sim.values.get(refs[0]))
            self.assertEqual(0, sim.values.get(refs[1]))


import sys
this = sys.modules[__name__]
createTestCases(this, 'TestReset', TestResetBase)
            
if __name__ == '__main__':
    unittest.main()
