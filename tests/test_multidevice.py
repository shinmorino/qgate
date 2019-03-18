from __future__ import print_function
from __future__ import absolute_import


from tests.test_base import *
from qgate.script import *
import numpy as np

if hasattr(qgate.simulator, 'cudaruntime') :

    class TestMultiDeviceCUDA(SimulatorTestBase) :

        def setUp(self) :
            #self.n_qregs = 3
            #self.n_lanes_in_chunk = 2
            self.n_qregs = 20
            self.n_lanes_in_chunk = 18
        
        def run_sim(self, circuit, multiDevice) :
            sim = qgate.simulator.cuda(isolate_circuits=False)

            n_lanes_in_chunk = -1
            device_ids = []
            if multiDevice :
                n_lanes_in_chunk = self.n_lanes_in_chunk
                device_ids = [0] * (1 << (self.n_qregs - self.n_lanes_in_chunk))
            
            sim.set_preference(n_lanes_in_chunk = n_lanes_in_chunk, device_ids = device_ids)
            sim.run(circuit)
            return sim

        def compare(self, circuit) :
            mc_states = self.run_sim(circuit, True).qubits.get_states()
            sc_states = self.run_sim(circuit, False).qubits.get_states()

            n_states = 1 << self.n_qregs
            self.assertTrue(np.allclose(mc_states, sc_states))

        def test_hadmard_gate(self) :
            circuit = new_circuit()
            qregs = new_qregs(self.n_qregs)
            circuit.add([h(qreg) for qreg in qregs])
            self.compare(circuit)

        def test_cx_gate(self) :
            circuit = new_circuit()
            qregs = new_qregs(self.n_qregs)
            circuit.add(x(qregs[0]))
            for idx in range(0, self.n_qregs - 1) :
                circuit.add(ctrl(qregs[idx]).x(qregs[idx + 1]))
            self.compare(circuit)

        def test_measure_x_mimimal(self) :
            circuit = new_circuit()
            qregs = new_qregs(self.n_qregs)
            init_cregs = new_references(self.n_qregs)
            neg_cregs = new_references(self.n_qregs)

            circuit.add(x(qregs[0]))
            circuit.add(measure(neg_cregs[1], qregs[1]))
            circuit.add(measure(neg_cregs[2], qregs[2]))

            sim = self.run_sim(circuit, True)
            self.assertEqual(sim.values.get(neg_cregs[1]), 0)

        def test_measure_x_minimal_2(self) :
            circuit = new_circuit()
            qregs = new_qregs(self.n_qregs)
            init_cregs = new_references(self.n_qregs)
            neg_cregs = new_references(1)

            for idx in range(0, self.n_qregs) :
                circuit.add(measure(init_cregs[idx], qregs[idx]))
            circuit.add(x(qregs[0]))
            circuit.add(measure(neg_cregs[0], qregs[1]))
            sim = self.run_sim(circuit, True)
            self.assertEqual(0, sim.values.get(neg_cregs[0]))

        def test_measure_x_minimal_3(self) :
            circuit = new_circuit()
            qregs = new_qregs(self.n_qregs)
            neg_cregs = new_references(self.n_qregs)

            circuit.add(x(qregs[2]))
            circuit.add(measure(neg_cregs[0], qregs[0]))
            circuit.add(measure(neg_cregs[1], qregs[1]))
            circuit.add(measure(neg_cregs[2], qregs[2]))
            sim = self.run_sim(circuit, True)
            
            self.assertEqual(0, sim.values.get(neg_cregs[0]))
            self.assertEqual(0, sim.values.get(neg_cregs[1]))
            self.assertEqual(1, sim.values.get(neg_cregs[2]))

        def test_measure_x(self) :
            for lane in range(0, self.n_qregs) :
                circuit = new_circuit()
                qregs = new_qregs(self.n_qregs)
                init_cregs = new_references(self.n_qregs)
                neg_cregs = new_references(self.n_qregs)

                for idx in range(0, self.n_qregs) :
                    circuit.add(measure(init_cregs[idx], qregs[idx]))
                circuit.add(x(qregs[lane]))
                for idx in range(0, self.n_qregs) :
                    circuit.add(measure(neg_cregs[idx], qregs[idx]))
                sim = self.run_sim(circuit, True)
                for idx in range(0, self.n_qregs) :
                    self.assertEqual(0, sim.values.get(init_cregs[idx]))
                    if idx == lane :
                        self.assertEqual(1, sim.values.get(neg_cregs[idx]))
                    else :
                        # print('lane {}, idx []'.format(lane, idx))
                        self.assertEqual(0, sim.values.get(neg_cregs[idx]))

                
if __name__ == '__main__':
    unittest.main()
