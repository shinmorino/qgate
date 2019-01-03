from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

class TestControlGateBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestControlGateBase:
            raise unittest.SkipTest()
        super(TestControlGateBase, cls).setUpClass()

    def run_sim(self) :
        return self._run_sim().get_qubits().get_states(qgate.simulator.prob)
    
    def test_cx_gate_2qubits(self) :
        new_program()
        qregs = allocate_qreg(2)            # |00>
        op(cx(qregs[0], qregs[1]))          # |00>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[0], 1)

        new_program()
        qregs = allocate_qreg(2)
        op(x(qregs[0]))                     # |01>
        op(cx(qregs[0], qregs[1]))          # |11>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[3], 1)

        new_program()
        qregs = allocate_qreg(2)
        op(x(qregs[1]))                     # |10>
        op(cx(qregs[0], qregs[1]))          # |10>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[2], 1)

        new_program()
        qregs = allocate_qreg(2)
        op(x(qregs))                         # |11>
        op(cx(qregs[0], qregs[1]))           # |01>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[1], 1)

        new_program()
        qregs = allocate_qreg(2)            # |00>
        op(cx(qregs[1], qregs[0]))          # |00>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[0], 1)

        new_program()
        qregs = allocate_qreg(2)
        op(x(qregs[0]))                     # |01>
        op(cx(qregs[1], qregs[0]))          # |01>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[1], 1)

        new_program()
        qregs = allocate_qreg(2)
        op(x(qregs[1]))                     # |10>
        op(cx(qregs[1], qregs[0]))          # |11>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[3], 1)

        new_program()
        qregs = allocate_qreg(2)
        op(x(qregs))                         # |11>
        op(cx(qregs[1], qregs[0]))           # |10>
        probs = self.run_sim()
        self.assertAlmostEqual(probs[2], 1)

    def test_cx_gate_multibits(self) :

        for n_qregs in range(1, 9) :
            for control in range(0, n_qregs) :
                for target in range(0, n_qregs) :
                    if control == target :
                        continue
                    
                    new_program()
                    qregs = allocate_qreg(n_qregs)
                    op(ID(qregs))
                    op(cx(qregs[control], qregs[target]))
                    probs = self.run_sim()
                    self.assertAlmostEqual(probs[0], 1)
                    
                    new_program()
                    qregs = allocate_qreg(n_qregs)
                    op(ID(qregs))
                    op(x(qregs[control]))
                    op(cx(qregs[control], qregs[target]))
                    probs = self.run_sim()
                    self.assertAlmostEqual(probs[(1 << control) | (1 << target)], 1)

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestControlGate', TestControlGateBase)

if __name__ == '__main__':
    unittest.main()
