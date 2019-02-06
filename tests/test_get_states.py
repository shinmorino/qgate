from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
import numpy as np


class TestGetStatesBase(SimulatorTestBase) :
    
    @classmethod
    def setUpClass(cls):
        if cls is TestGetStatesBase:
            raise unittest.SkipTest()
        super(TestGetStatesBase, cls).setUpClass()

    def setUp(self) :
        self.n_qregs = 4
        
    def run_sim(self, circuit, isolate_circuits = False) :
        return self._run_sim(circuit, isolate_circuits)

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
    
    def test_initial_states(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([a(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states = sim.qubits.get_states()
        self.assertEqual(1 + 0j, states[0])
        self.assertTrue(all(states[1:-1] == 0.))
        
    def test_initial_states_with_offset(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([a(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states = sim.qubits.get_states(key = slice(1, None))
        self.assertTrue(all(states[0:-1] == 0.))
        
    def test_x(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([x(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states = sim.qubits.get_states()
        self.assertEqual(1, states[-1])
        self.assertTrue(all(states[0:-2] == 0.))
        
        states = sim.qubits.get_states(key = slice(0, -1))
        self.assertTrue(all(states == 0.))

        state = sim.qubits.get_states(key = -1)
        self.assertEqual(1, state)
        
    def test_h(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states = sim.qubits.get_states(mathop = qgate.simulator.prob)
        self.assertTrue(np.allclose(1 / len(states), states))

    def test_inversed_range(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, False)
        states = sim.qubits.get_states(key = slice(10, 0))
        self.assertEqual(0, len(states))

        states = sim.qubits.get_states(key = slice(10, 0, -1))
        self.assertEqual(10, len(states))

    def test_step(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, False)
        states = sim.qubits.get_states()

        states_test = sim.qubits.get_states(key = slice(0, 10, 3))
        self.assertEqual(4, len(states_test))
        states_ref = states[:10:3]
        self.assertTrue(np.allclose(states_ref, states_test))
        
        states_test = sim.qubits.get_states(key = slice(0, 10, 7))
        self.assertEqual(2, len(states_test))
        states_ref = states[:10:7]
        self.assertTrue(np.allclose(states_ref, states_test))

    def test_negative_step(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, False)
        states = sim.qubits.get_states()

        states_test = sim.qubits.get_states(key = slice(10, None, -3))
        self.assertEqual(4, len(states_test))
        states_ref = states[10::-3]
        self.assertTrue(np.allclose(states_ref, states_test))
        
        states_test = sim.qubits.get_states(key = slice(10, None, -7))
        self.assertEqual(2, len(states_test))
        states_ref = states[10::-7]
        self.assertTrue(np.allclose(states_ref, states_test))
        
    def test_getter(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states_getter = sim.qubits.states[:]
        states = sim.qubits.get_states()
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = sim.qubits.states[10:]
        states = sim.qubits.get_states(key=slice(10, None) )
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = sim.qubits.states[:10]
        states = sim.qubits.get_states(key=slice(None, 10))
        self.assertTrue(np.all(states == states_getter))
        
        
    def test_getter_inversed(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states_getter = sim.qubits.states[::-1]
        states = sim.qubits.get_states()
        states = states[::-1]
        self.assertEqual(len(states), len(states_getter))
        self.assertTrue(np.all(states == states_getter))
     
        states_getter = sim.qubits.states[10::-1]
        states = sim.qubits.get_states(key=slice(0, 12))
        states = states[-1:0:-1]
        self.assertEqual(len(states), len(states_getter))
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = sim.qubits.states[-1:10:-1]
        states = sim.qubits.get_states(key=slice(10, None))
        states = states[-1:0:-1]
        #self.assertTrue(np.all(states == states_getter))
        
    def test_prob_getter(self) :
        circuit = new_circuit()
        qregs = new_qregs(self.n_qregs)
        circuit.add([h(qreg) for qreg in qregs])
        sim = self.run_sim(circuit, True)
        states_getter = sim.qubits.prob[:]
        states = sim.qubits.get_states(qgate.simulator.prob)
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = sim.qubits.prob[10:]
        states = sim.qubits.get_states(qgate.simulator.prob, key=slice(10, None))
        self.assertTrue(np.all(states == states_getter))
        
        states_getter = sim.qubits.prob[:10]
        states = sim.qubits.get_states(qgate.simulator.prob, key=slice(None, 10))
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
