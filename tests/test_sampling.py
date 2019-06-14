from __future__ import print_function
from __future__ import absolute_import

import unittest
import qgate
from qgate.script import *
import numpy as np

class TestSampling(unittest.TestCase) :

    def test_observation(self) :
        refs = new_references(3)
        obs = qgate.simulator.observation.Observation(refs, 0b100, 0b001)
        self.assertEqual(repr(obs), '10*')
        self.assertEqual(obs.int, 4)

        self.assertIs(obs(refs[0]), None)
        self.assertEqual(obs(refs[1]), 0)
        self.assertEqual(obs(refs[2]), 1)

    def test_observation_eq(self) :
        refs = new_references(3)
        obs0 = qgate.simulator.observation.Observation(refs, 0b100, 0b001)
        self.assertTrue(obs0 == obs0)
        
        obs1 = qgate.simulator.observation.Observation(refs, 0b100, 0b000)
        self.assertFalse(obs0 == obs1)

        obs1 = qgate.simulator.observation.Observation(refs[0:1], 0b100, 0b001)
        self.assertFalse(obs0 == obs1)
        
        obs1 = qgate.simulator.observation.Observation(refs, 0b110, 0b001)
        self.assertFalse(obs0 == obs1)

    def test_observation_lt(self) :
        refs = new_references(3)
        obs = qgate.simulator.observation.Observation(refs, 0b100, 0b001)
        with self.assertRaises(TypeError) :
            v = obs < obs
        with self.assertRaises(TypeError) :
            v = obs < self

    def test_observation_list(self) :
        refs = new_references(3)
        obs = qgate.simulator.observation.Observation(refs, 0b100, 0b001)
        with self.assertRaises(TypeError) :
            v = obs < obs
        with self.assertRaises(TypeError) :
            v = obs < self
    
    def test_sampling_one_shot(self) :
        qregs = new_qregs(4)
        cregs = new_references(4)
        circuit = [I(qreg) for qreg in qregs]

        sim = qgate.simulator.py()
        sim.run(circuit)
        obs = sim.obs(cregs)
        self.assertEqual(int(obs), 0)
    
    def test_sampling_128_shots(self) :
        qregs = new_qregs(4)
        cregs = new_references(4)
        circuit = [[I(qreg) for qreg in qregs],
                   X(qregs[0]), 
                   [measure(creg, qreg) for creg, qreg in zip(cregs[0:-1], qregs[0:-1])]]

        sim = qgate.simulator.cpu()
        obslist = sim.sample(circuit, cregs, 128)
        # sample list length
        self.assertEqual(len(obslist), 128)
        # sampled value
        obs = obslist[0]
        self.assertTrue(isinstance(obs, qgate.simulator.observation.Observation))
        self.assertEqual(repr(obs), "*001")

        # intarray
        values = obslist.intarray
        self.assertEqual(len(values), 128)
        self.assertTrue(np.all(values == 1))

        # __call__ to extract
        # creg[0] -> 1
        extracted = obslist(cregs[0])
        self.assertEqual(len(extracted), 128)
        b = [v == 1 for v in extracted]
        self.assertTrue(np.all(b))
        # creg[1] -> 0
        extracted = obslist(cregs[1])
        b = [v == 0 for v in extracted]
        self.assertTrue(np.all(b))
        # creg[3] -> None
        extracted = obslist(cregs[3])
        b = [v is None for v in extracted]
        self.assertTrue(np.all(b))

        # __getitem__
        half = obslist[0:64]
        self.assertEqual(len(half), 64)

    def test_observation_list_getitem(self) :
        cregs = new_references(4)
        values = np.random.random((64))
        obslist = qgate.simulator.observation.ObservationList(cregs, values, 0)
        obslist32 = obslist[:32]
        self.assertEqual(obslist32._reflist, obslist._reflist)
        self.assertTrue(np.allclose(obslist32._values, obslist._values[:32]))

    def test_observation_histgram(self) :
        qregs = new_qregs(4)
        cregs = new_references(4)
        circuit = [[I(qreg) for qreg in qregs],
                   X(qregs[0]), 
                   [measure(creg, qreg) for creg, qreg in zip(cregs[0:-1], qregs[0:-1])]]

        sim = qgate.simulator.cpu()
        obslist = sim.sample(circuit, cregs, 128)

        hist = obslist.histgram()
        self.assertEqual(hist.n_samples, 128)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0], 0)
        self.assertEqual(hist[1], 128)

        keys = [key for key in hist.keys()]
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], 1)

        values = [value for value in hist.values()]
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 128)

        copied = dict(hist.items())
        self.assertEqual(copied, hist._hist)
        

if __name__ == '__main__':
    unittest.main()
