from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
import numpy as np


class DummySamplingPool :
    def __init__(self, prob, empty_lanes, qreg_ordering) :
        self.prob = prob
        self.empty_lanes = empty_lanes
        self.qreg_ordering = qreg_ordering


class TestSamplingPoolBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestSamplingPoolBase:
            raise unittest.SkipTest()
        super(TestSamplingPoolBase, cls).setUpClass()

    def setUp(self) :
        self.n_qubits = 10
        self.n_samples = 1024
        if self.runtime == 'cpu' or self.runtime == 'cuda' :
            self.n_qubits = 20

    def run_sim(self, circuit) :
        return self._run_sim(circuit)

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
        
    def test_create_sampling_pool(self) :
        qregs = new_qregs(self.n_qubits)
        circuit = [H(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)
        sp = sim.qubits.create_sampling_pool(qregs)
        del sp
        
    def test_empty_lanes(self) :
        qreg_ordering = new_qregs(self.n_qubits)
        qregs = qreg_ordering[::2]
        empty = qreg_ordering[1::2]
        
        circuit = [H(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)
        
        sp = sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)
        self.assertEqual(sp.qreg_ordering, qreg_ordering)

        for qreg in empty :
            self.assertTrue(qreg_ordering.index(qreg) in sp.empty_lanes)
            
    def test_H_prob(self) :
        qregs = new_qregs(self.n_qubits)
        circuit = [H(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)
        
        sp = sim.qubits.create_sampling_pool(qregs, DummySamplingPool)
        n_states = 1 << self.n_qubits
        self.assertEqual(len(sp.prob), n_states)
        self.assertTrue(np.allclose(sp.prob, 1. / n_states))

    def test_X_prob(self) :
        qregs = new_qregs(self.n_qubits)
        for xlane in range(self.n_qubits) :
            circuit = list()
            circuit += [H(qreg) for qreg in qregs[:xlane]]
            circuit += [X(qregs[xlane])]
            circuit += [H(qreg) for qreg in qregs[xlane+1:]]
            sim = self.run_sim(circuit)
            sp = sim.qubits.create_sampling_pool(qregs, DummySamplingPool)
            idxlist = [idx for idx in range(self.n_samples) if (idx & (1 << xlane)) == 0]
            # print(xlane, idxlist, sp.prob)
            self.assertTrue(np.allclose(sp.prob[idxlist], 0.))

    def test_dup_qreg(self) :
        qregs = new_qregs(self.n_qubits)
        circuit = [H(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)

        qreg_ordering = qregs + qregs
        with self.assertRaises(RuntimeError):
            sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)

    def test_qreg_reordering(self) :
        qregs = new_qregs(self.n_qubits)
        xqreg = qregs[0]
        hqregs = qregs[1:]
        
        circuit = list()
        circuit.append(X(xqreg))
        circuit += [H(qreg) for qreg in hqregs]
        sim = self.run_sim(circuit)

        for xlane in range(9) :
            qreg_ordering = hqregs[:xlane] + [xqreg] + hqregs[xlane:]
            sp = sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)
            idxlist = [idx for idx in range(self.n_samples) if (idx & (1 << xlane)) == 0]
            # print(xlane, idxlist, sp.prob)
            self.assertTrue(np.allclose(sp.prob[idxlist], 0.))

    def test_hidden_qregs(self) :
        qregs = new_qregs(self.n_qubits)
        circuit = [H(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)

        qreg_ordering = qregs[::2]
        sp = sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)
        n_states = 1 << (self.n_qubits // 2)
        self.assertEqual(len(sp.prob), n_states)
        if not np.allclose(sp.prob, 1. / n_states) :
            print(sp.prob)
        self.assertTrue(np.allclose(sp.prob, 1. / n_states))

        qreg_ordering = qregs[1::2]
        sp = sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)
        self.assertEqual(len(sp.prob), n_states)
        self.assertTrue(np.allclose(sp.prob, 1. / n_states))

    def test_hidden_qregs_2(self) :
        qregs = new_qregs(self.n_qubits)
        circuit = [H(qreg) for qreg in qregs[::2]]
        circuit += [X(qreg) for qreg in qregs[1::2]]
        sim = self.run_sim(circuit)

        if False :
            print(type(self))
            sim.qubits.set_ordering(qregs)
            print(sim.qubits.prob[:])

        qreg_ordering = qregs[::2]
        sp = sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)
        n_states = 1 << (self.n_qubits // 2)
        self.assertEqual(len(sp.prob), n_states)
        if not np.allclose(sp.prob, 1. / n_states) :
            print(sp.prob)
        self.assertTrue(np.allclose(sp.prob, 1. / n_states))

        #qreg_ordering = qregs[1::2]
        #sp = sim.qubits.create_sampling_pool(qreg_ordering, DummySamplingPool)
        #self.assertEqual(len(sp.prob), n_states)
        #self.assertTrue(np.allclose(sp.prob, 1. / n_states))
        
    def test_sampling_pool_sample(self) :
        qregs = new_qregs(self.n_qubits)
        circuit = [H(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)
        sp = sim.qubits.create_sampling_pool(qregs)
        obs = sp.sample(self.n_samples)
        self.assertEqual(len(obs), self.n_samples)
        del sp
        
    def test_sampling_pool_empty_lane(self) :
        qreg_ordering = new_qregs(self.n_qubits)
        qregs = qreg_ordering[::2]
        circuit = [X(qreg) for qreg in qregs]
        sim = self.run_sim(circuit)
        sp = sim.qubits.create_sampling_pool(qreg_ordering)
        obs = sp.sample(self.n_samples)
        for qreg in qreg_ordering :
            if qreg in qregs :
                self.assertTrue(np.all(np.array(obs(qreg)) == 1))
            else :
                self.assertTrue(all(v is None for v in obs(qreg)))

        arr = obs.intarray
        for qreg in qreg_ordering :
            pos = qreg_ordering.index(qreg)
            mask = 1 << pos
            if qreg in qregs :
                self.assertTrue(np.all((arr & mask) != 0))
            else :
                self.assertTrue(np.all((arr & mask) == 0))

        # FIXME: add one_static tests

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestSamplingPool', TestSamplingPoolBase)

if __name__ == '__main__':
    unittest.main()
