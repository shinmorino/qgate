from __future__ import print_function
from __future__ import absolute_import


from tests.test_base import *
from qgate.script import *
import numpy as np

if hasattr(qgate.simulator, 'cudaruntime') :

    class TestMultiDeviceCUDA(SimulatorTestBase) :

        def set_mgpu_preference(self) :
            # max chunk size, 2 MB.
            max_po2idx_per_chunk = 21
            # device memory per memstore
            memory_store_size = 5 * (1 << 20)
            # device ids.
            device_ids = [0] * 8
            # initialize
            qgate.simulator.cudaruntime.set_preference(device_ids, max_po2idx_per_chunk, memory_store_size)
        
        def term_module(self) :
            qgate.simulator.cudaruntime.module_finalize()

        def setUp(self) :
            # using fp64, 16 MB.
            self.n_qregs = 20
            self.term_module()

        def tearDown(self) :
            self.term_module()
            qgate.simulator.cudaruntime.reset_preference()

        def run_sim(self, circuit) :
            sim = qgate.simulator.cuda(isolate_circuits=False)
            sim.run(circuit)
            return sim

        def compare(self, circuit) :
            self.set_mgpu_preference();
            mc_states = self.run_sim(circuit).qubits.get_states()
            self.term_module()
            qgate.simulator.cudaruntime.reset_preference()
            sc_states = self.run_sim(circuit).qubits.get_states()
            self.term_module()

            n_states = 1 << self.n_qregs
            self.assertTrue(np.allclose(mc_states, sc_states))

        def test_hadmard_gate(self) :
            qregs = new_qregs(self.n_qregs)
            circuit = [H(qreg) for qreg in qregs]
            self.compare(circuit)

        def test_cx_gate(self) :
            qregs = new_qregs(self.n_qregs)
            circuit = [ X(qregs[0]) ]
            for idx in range(0, self.n_qregs - 1) :
                cx = ctrl(qregs[idx]).X(qregs[idx + 1])
                circuit.append(cx)
            self.compare(circuit)

        def test_measure_x_mimimal(self) :
            self.set_mgpu_preference()
            qregs = new_qregs(self.n_qregs)
            neg_cregs = new_references(self.n_qregs)

            circuit = [
                [I(qreg) for qreg in qregs], 
                X(qregs[0]), 
                measure(neg_cregs[1], qregs[1]),
                measure(neg_cregs[2], qregs[2])
            ]

            sim = self.run_sim(circuit)
            self.assertEqual(sim.values.get(neg_cregs[1]), 0)

        def test_measure_x_minimal_2(self) :
            self.set_mgpu_preference()
            qregs = new_qregs(self.n_qregs)
            init_cregs = new_references(self.n_qregs)
            neg_cregs = new_references(1)

            circuit = []
            for idx in range(0, self.n_qregs) :
                circuit.append(measure(init_cregs[idx], qregs[idx]))
            circuit.append(X(qregs[0]))
            circuit.append(measure(neg_cregs[0], qregs[1]))
            sim = self.run_sim(circuit)
            self.assertEqual(0, sim.values.get(neg_cregs[0]))

        def test_measure_x_minimal_3(self) :
            self.set_mgpu_preference()
            qregs = new_qregs(self.n_qregs)
            neg_cregs = new_references(self.n_qregs)

            circuit = [
                [I(qreg) for qreg in qregs], 
                X(qregs[2]),
                measure(neg_cregs[0], qregs[0]),
                measure(neg_cregs[1], qregs[1]),
                measure(neg_cregs[2], qregs[2])
            ]
            sim = self.run_sim(circuit)
            
            self.assertEqual(0, sim.values.get(neg_cregs[0]))
            self.assertEqual(0, sim.values.get(neg_cregs[1]))
            self.assertEqual(1, sim.values.get(neg_cregs[2]))

        def test_measure_x(self) :
            self.set_mgpu_preference()
            for lane in range(0, self.n_qregs) :
                qregs = new_qregs(self.n_qregs)
                init_cregs = new_references(self.n_qregs)
                neg_cregs = new_references(self.n_qregs)

                circuit = []
                for idx in range(0, self.n_qregs) :
                    circuit += [measure(init_cregs[idx], qregs[idx])]
                circuit.append(X(qregs[lane]))
                for idx in range(0, self.n_qregs) :
                    circuit += [measure(neg_cregs[idx], qregs[idx])]
                sim = self.run_sim(circuit)
                for idx in range(0, self.n_qregs) :
                    self.assertEqual(0, sim.values.get(init_cregs[idx]))
                    if idx == lane :
                        self.assertEqual(1, sim.values.get(neg_cregs[idx]))
                    else :
                        # print('lane {}, idx []'.format(lane, idx))
                        self.assertEqual(0, sim.values.get(neg_cregs[idx]))

                
if __name__ == '__main__':
    unittest.main()
