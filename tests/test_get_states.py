from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.qasm.qelib1 import *
from qgate.qasm.script import *


class TestGetStatesBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestGetStatesBase:
            raise unittest.SkipTest()
        super(TestGetStatesBase, cls).setUpClass()

    def setUp(self) :
        self.n_qregs = 4
        
    def run_sim(self, isolate_circuits = False) :
        sim = self._run_sim(isolate_circuits)
        return sim.get_qubits()

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_initial_states(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(a(qreg))
        qubits = self.run_sim(True)
        states = qubits.get_states()
        self.assertEqual(1 + 0j, states[0])
        self.assertTrue(all(states[1:-1] == 0.))
        
    def test_initial_states_with_offset(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(a(qreg))
        qubits = self.run_sim(True)
        states = qubits.get_states(key = slice(1, None))
        self.assertTrue(all(states[0:-1] == 0.))
        
    def test_x(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(x(qreg))
        qubits = self.run_sim(True)
        states = qubits.get_states()
        self.assertEqual(1, states[-1])
        self.assertTrue(all(states[0:-2] == 0.))
        
        states = qubits.get_states(key = slice(0, -1))
        self.assertTrue(all(states == 0.))

        state = qubits.get_states(key = -1)
        self.assertEqual(1, state)
        
    def test_h(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(True)
        states = qubits.get_states(mathop = qgate.simulator.prob)
        self.assertTrue(np.allclose(1 / len(states), states))

    def test_inversed_range(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(False)
        states = qubits.get_states(key = slice(10, 0))
        self.assertEqual(0, len(states))

        states = qubits.get_states(key = slice(10, 0, -1))
        self.assertEqual(10, len(states))

    def test_step(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(False)
        states = qubits.get_states()

        states_test = qubits.get_states(key = slice(0, 10, 3))
        self.assertEqual(4, len(states_test))
        states_ref = states[:10:3]
        self.assertTrue(np.allclose(states_ref, states_test))
        
        states_test = qubits.get_states(key = slice(0, 10, 7))
        self.assertEqual(2, len(states_test))
        states_ref = states[:10:7]
        self.assertTrue(np.allclose(states_ref, states_test))

    def test_negative_step(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(False)
        states = qubits.get_states()

        states_test = qubits.get_states(key = slice(10, None, -3))
        self.assertEqual(4, len(states_test))
        states_ref = states[10::-3]
        self.assertTrue(np.allclose(states_ref, states_test))
        
        states_test = qubits.get_states(key = slice(10, None, -7))
        self.assertEqual(2, len(states_test))
        states_ref = states[10::-7]
        self.assertTrue(np.allclose(states_ref, states_test))
        
    def test_getter(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(True)
        states_getter = qubits.states[:]
        states = qubits.get_states()
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = qubits.states[10:]
        states = qubits.get_states(key=slice(10, None) )
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = qubits.states[:10]
        states = qubits.get_states(key=slice(None, 10))
        self.assertTrue(np.all(states == states_getter))
        
        
    def test_getter_inversed(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(True)
        states_getter = qubits.states[::-1]
        states = qubits.get_states()
        states = states[::-1]
        self.assertEqual(len(states), len(states_getter))
        self.assertTrue(np.all(states == states_getter))
     
        states_getter = qubits.states[10::-1]
        states = qubits.get_states(key=slice(0, 12))
        states = states[-1:0:-1]
        self.assertEqual(len(states), len(states_getter))
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = qubits.states[-1:10:-1]
        states = qubits.get_states(key=slice(10, None))
        states = states[-1:0:-1]
        #self.assertTrue(np.all(states == states_getter))
        
    def test_prob_getter(self) :
        new_program()
        qreg = allocate_qreg(self.n_qregs)
        op(h(qreg))
        qubits = self.run_sim(True)
        states_getter = qubits.prob[:]
        states = qubits.get_states(qgate.simulator.prob)
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = qubits.prob[10:]
        states = qubits.get_states(qgate.simulator.prob, key=slice(10, None))
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = qubits.prob[:10]
        states = qubits.get_states(qgate.simulator.prob, key=slice(None, 10))
        self.assertTrue(np.all(states == states_getter))
        
                    
import sys
this = sys.modules[__name__]

def setUp(self) :
    self.n_qregs = 20

createTestCases(this, 'TestGetStates', TestGetStatesBase)

class TestGetStates_20bits_CPU(TestGetStatesCPU) :
    def setUp(self) :
        self.n_qregs = 20

if hasattr(qgate.simulator, 'cudaruntime') :
    class TestGetStates_20bits_CUDA(TestGetStatesCUDA) :
        def setUp(self) :
            self.n_qregs = 20



if __name__ == '__main__':
    unittest.main()
