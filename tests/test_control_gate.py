from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *

class TestControlGateBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestControlGateBase:
            raise unittest.SkipTest()
        super(TestControlGateBase, cls).setUpClass()

    def run_sim(self, circuit, qreg_ordering) :
        sim = self.create_simulator()
        sim.qubits.set_ordering(qreg_ordering)
        sim.run(circuit)
        return sim.qubits.prob[:]
    
    def test_cx_gate_2qubits(self) :
        qregs = new_qregs(2)                     # |00>
        circuit = ctrl(qregs[0]).X(qregs[1])     # |00>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[0])

        circuit = [ X(qregs[0]),                 # |01>
                    ctrl(qregs[0]).X(qregs[1]) ] # |11>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[3])

        circuit = [ X(qregs[1]),                 # |10>
                    ctrl(qregs[0]).X(qregs[1]) ] # |10>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[2])

        circuit = [ [X(qreg) for qreg in qregs], # |11>
                    ctrl(qregs[0]).X(qregs[1]) ] # |01>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[1])

        circuit = [ ctrl(qregs[1]).X(qregs[0]) ] # |00>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[0])

        circuit = [ X(qregs[0]),                 # |01>
                    ctrl(qregs[1]).X(qregs[0]) ] # |01>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[1])

        circuit = [ X(qregs[1]),                 # |10>
                    ctrl(qregs[1]).X(qregs[0]) ] # |11>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[3])

        circuit = [ [X(qreg) for qreg in qregs], # |11>
                    ctrl(qregs[1]).X(qregs[0]) ] # |10>
        probs = self.run_sim(circuit, qregs)
        self.assertAlmostEqual(1, probs[2])

    def test_cx_gate_multibits(self) :

        for n_qregs in range(1, 9) :
            for control in range(0, n_qregs) :
                for target in range(0, n_qregs) :
                    if control == target :
                        continue
                    
                    qregs = new_qregs(n_qregs)
                    circuit = [ [I(qreg) for qreg in qregs],
                                ctrl(qregs[control]).X(qregs[target]) ]
                    probs = self.run_sim(circuit, qregs)
                    self.assertAlmostEqual(1, probs[0])
                    
                    circuit = [ [I(qreg) for qreg in qregs],
                                X(qregs[control]),
                                ctrl(qregs[control]).X(qregs[target]) ]
                    probs = self.run_sim(circuit, qregs)
                    ext_idx = (1 << control) | (1 << target)
                    self.assertAlmostEqual(1, probs[ext_idx])

    def test_cx_gate_multibits_2(self) :

        n_qregs = 3
        control = 2
        target = 0

        qregs = new_qregs(n_qregs)
        circuit = [ [I(qreg) for qreg in qregs],
                    X(qregs[control]),
                    ctrl(qregs[control]).X(qregs[target]) ]
        sim = self._run_sim(circuit, qregs)
        probs = sim.qubits.get_states(qgate.simulator.prob)
        ext_idx = (1 << control) | (1 << target)
        self.assertAlmostEqual(1, probs[ext_idx])
                
    def test_n_bit_controlled_x_gate(self) :

        for n_qregs in range(2, 10) :
            qregs = new_qregs(n_qregs)
            circuit = [ [I(qreg) for qreg in qregs], 
                        [X(qreg) for qreg in qregs[0:-1]],
                        ctrl(qregs[0:-1]).X(qregs[-1]) ]
            probs = self.run_sim(circuit, qregs)
            self.assertAlmostEqual(1 << n_qregs, len(probs))
            self.assertAlmostEqual(1, probs[-1])
            
    def test_n_bit_controlled_x_gate_2(self) :

        for n_qregs in range(3, 10) :
            qregs = new_qregs(n_qregs)
            circuit = [ I(qregs[0]),
                        [X(qreg) for qreg in qregs[1:-1]],
                        ctrl(qregs[0:-1]).X(qregs[-1]) ]
            probs = self.run_sim(circuit, qregs)
            self.assertAlmostEqual(1 << n_qregs, len(probs))
            self.assertAlmostEqual(0, probs[-1])
        

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestControlGate', TestControlGateBase)

if __name__ == '__main__':
    unittest.main()
