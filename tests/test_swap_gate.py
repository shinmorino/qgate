from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *

class TestSwapGateBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestSwapGateBase:
            raise unittest.SkipTest()
        super(TestSwapGateBase, cls).setUpClass()

    def run_sim(self, circuit, qreg_ordering) :
        sim = self.create_simulator()
        sim.qubits.set_ordering(qreg_ordering)
        sim.run(circuit)
        return sim.qubits.prob[:]
    
    def test_cx_gate_2qubits(self) :
        ordering = new_qregs(2)
        qreg0, qreg1 = ordering

        circuit = Swap(qreg0, qreg1)           # |00>
        probs = self.run_sim(circuit, ordering)
        self.assertAlmostEqual(1, probs[0])

        circuit = [ X(qreg0),                  # |01>
                    Swap(qreg0, qreg1) ]       # |10>
        probs = self.run_sim(circuit, ordering)
        self.assertAlmostEqual(1, probs[2])

        circuit = [ X(qreg1),                  # |10>
                    Swap(qreg0, qreg1) ]       # |01>
        probs = self.run_sim(circuit, ordering)
        self.assertAlmostEqual(1, probs[1])

        circuit = [ X(qreg0), X(qreg1),        # |11>
                    Swap(qreg0, qreg1) ]       # |11>
        probs = self.run_sim(circuit, ordering)
        self.assertAlmostEqual(1, probs[3])

    def test_swap_gate_multibits(self) :

        for n_qregs in range(3, 9) :
            qregs = new_qregs(n_qregs)
            circuit = []
            circuit.append(X(qregs[0])) # |0...01>
                        
            for idx in range(0, n_qregs - 1) :
                circuit.append(Swap(qregs[idx], qregs[idx + 1]))

            # test each bit
            probs = self.run_sim(circuit, qregs)
            for idx in range(0, n_qregs - 1) :
                ext_idx = 1 << idx
                self.assertAlmostEqual(0, probs[ext_idx])
            ext_idx = 1 << (n_qregs - 1)
            self.assertAlmostEqual(1, probs[ext_idx])
        

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestSwapGate', TestSwapGateBase)

if __name__ == '__main__':
    unittest.main()
