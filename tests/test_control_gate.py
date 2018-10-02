from __future__ import print_function
from __future__ import absolute_import

import qgate
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

import unittest

class TestControlGate(unittest.TestCase) :

    def run_sim(self) :
        program = current_program()
        program = qgate.qasm.process(program, isolate_circuits=False)
        sim = qgate.simulator.cpu(program)
        sim.prepare()
        sim.run()
        return sim.get_qubits().get_probabilities()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
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

        for n_qregs in range(1, 11) :
            for control in range(0, n_qregs) :
                for target in range(0, n_qregs) :
                    if control == target :
                        continue
                    
                    new_program()
                    qregs = allocate_qreg(n_qregs)
                    op(cx(qregs[control], qregs[target]))
                    probs = self.run_sim()
                    self.assertAlmostEqual(probs[0], 1)
                    
                    new_program()
                    qregs = allocate_qreg(n_qregs)
                    op(x(qregs[control]))
                    op(cx(qregs[control], qregs[target]))
                    probs = self.run_sim()
                    self.assertAlmostEqual(probs[(1 << control) | (1 << target)], 1)


if __name__ == '__main__':
    unittest.main()
