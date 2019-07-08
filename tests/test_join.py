from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
import numpy as np

class TestJoinBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestJoinBase:
            raise unittest.SkipTest()
        super(TestJoinBase, cls).setUpClass()

    def setUp(self) :
        if self.runtime == 'py' :
            self.n_qregs = 8
        elif self.runtime == 'cpu' :
            self.n_qregs = 15
        elif self.runtime == 'cuda' :
            self.n_qregs = 20
        
    def run_sim(self, circuit, qreg_ordering) :
        sim = self.create_simulator()
        sim.qubits.set_ordering(qreg_ordering)
        sim.run(circuit)
        return sim.qubits.prob[:]

    def test_join_1_by_1(self) :
        for n_qregs in range(2, self.n_qregs) :
            qregs = new_qregs(n_qregs)
            circuit = [H(qreg) for qreg in qregs]
            circuit += [ctrl(qregs[idx]).X(qregs[idx + 1]) for idx in range(len(qregs) - 1)]

            probs = self.run_sim(circuit, qregs)
            n_states = 1 << len(qregs)
            self.assertTrue(np.allclose(1 / n_states, probs[:]))

    def test_join_n_n(self) :
        for n_qregs in range(1, self.n_qregs // 2) :
            qregs_0 = new_qregs(n_qregs)
            qregs_1 = new_qregs(n_qregs)
            circuit = [ctrl(qregs_0[idx]).I(qregs_0[idx + 1]) for idx in range(len(qregs_0) - 1)]
            circuit += [H(qreg) for qreg in qregs_0]
            circuit += [ctrl(qregs_1[idx]).I(qregs_1[idx + 1]) for idx in range(len(qregs_1) - 1)]
            circuit += [H(qreg) for qreg in qregs_1]
            circuit += [ctrl(qregs_0[:-1]).I(qregs_1[0])]

            probs = self.run_sim(circuit, qregs_0 + qregs_1)
            n_states = 1 << (len(qregs_0) + len(qregs_1))
            self.assertTrue(np.allclose(1 / n_states, probs[:]))

    def test_join_n_1(self) :
        qregs = new_qregs(self.n_qregs)
        for target in qregs :
            circuit = [H(qreg) for qreg in qregs]
            controls = list(qregs)
            controls.remove(target)
            circuit += [ctrl(controls).I(target)]

            probs = self.run_sim(circuit, qregs)
            n_states = 1 << len(qregs)
            self.assertTrue(np.allclose(1 / n_states, probs[:]))

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestJoin', TestJoinBase)

if __name__ == '__main__':
    unittest.main()
