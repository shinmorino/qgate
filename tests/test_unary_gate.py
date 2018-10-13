from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

class TestUnaryGateBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestUnaryGateBase:
            raise unittest.SkipTest()
        super(TestUnaryGateBase, cls).setUpClass()
        
    def run_sim(self) :
        sim = self._run_sim()
        return sim.get_qubits().get_probabilities()
    
    def test_id_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(a(qregs))
        probs = self.run_sim()
        self.assertEqual(probs[0], 1.)
        
    def test_pauli_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(x(qregs))
        probs = self.run_sim()
        self.assertEqual(probs[1], 1.)
        
    def test_pauli_gate_2(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(x(qregs), x(qregs))
        probs = self.run_sim()
        self.assertEqual(probs[0], 1.)
        
    def test_hadmard_gate(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(h(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)
        
    def test_hadmard_gate2(self) :
        new_program()
        qregs = allocate_qreg(1)
        op(h(qregs), h(qregs))
        probs = self.run_sim()
        self.assertAlmostEqual(probs[0], 1)
        self.assertAlmostEqual(probs[1], 0.)

    def test_pauli_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            new_program()
            qregs = allocate_qreg(n_qubits)
            op(x(qregs))
            probs = self.run_sim()
            self.assertAlmostEqual(probs[(1 << n_qubits) - 1], 1)

    def test_hadamard_gate_multi_qubits(self) :
        for n_qubits in range(1, 11) :
            new_program()
            qregs = allocate_qreg(n_qubits)
            op(h(qregs))
            probs = self.run_sim()
            n_states = 1 << n_qubits
            for idx in range(n_states) :
                self.assertAlmostEqual(probs[idx], 1. / n_states)


class TestUnaryGatePy(TestUnaryGateBase) :
    def create_simulator(self, program) :
        return qgate.simulator.py(program)

class TestUnaryGateCPU(TestUnaryGateBase) :
    def create_simulator(self, program) :
        return qgate.simulator.cpu(program)
        
if hasattr(qgate.simulator, 'cudaruntime') :
    class TestUnaryGateCUDA(TestUnaryGateBase) :
        def create_simulator(self, program) :
            return qgate.simulator.cuda(program)
                
if __name__ == '__main__':
    unittest.main()
