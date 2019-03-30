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

    def run_sim(self, circuit) :
        sim = self._run_sim(circuit)
        probs = sim.qubits.get_states(qgate.simulator.prob)
        return sim.qubits, probs
    
    def test_cx_gate_2qubits(self) :
        qreg0, qreg1 = new_qregs(2)

        circuit = Swap(qreg0, qreg1)           # |00>
        qubits, probs = self.run_sim(circuit)
        self.assertAlmostEqual(1, probs[0])

        circuit = [ X(qreg0),                  # |01>
                    Swap(qreg0, qreg1) ]       # |10>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.lanes.get_state_index(qreg1)
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = [ X(qreg1),                  # |10>
                    Swap(qreg0, qreg1) ]       # |01>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.lanes.get_state_index(qreg0)
        self.assertAlmostEqual(1, probs[state_idx])

        circuit = [ X(qreg0), X(qreg1),        # |11>
                    Swap(qreg0, qreg1) ]       # |11>
        qubits, probs = self.run_sim(circuit)
        state_idx = qubits.lanes.get_state_index(qreg0, qreg1)
        self.assertAlmostEqual(1, probs[state_idx])
        

    def test_swap_gate_multibits(self) :

        for n_qregs in range(3, 9) :
            qregs = new_qregs(n_qregs)
            circuit = []
            circuit.append(X(qregs[0])) # |0...01>
                        
            for idx in range(0, n_qregs - 1) :
                circuit.append(Swap(qregs[idx], qregs[idx + 1]))

            # test each bit
            qubits, probs = self.run_sim(circuit)
            for idx in range(0, n_qregs - 1) :
                ext_idx = qubits.lanes.get_state_index(qregs[idx])
                self.assertAlmostEqual(0, probs[ext_idx])
            ext_idx = qubits.lanes.get_state_index(qregs[n_qregs - 1])
            self.assertAlmostEqual(1, probs[ext_idx])
        

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestSwapGate', TestSwapGateBase)

if __name__ == '__main__':
    unittest.main()
