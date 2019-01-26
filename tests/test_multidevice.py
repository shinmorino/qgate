from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append('C:\\projects\\qgate_sandbox')

from tests.test_base import *
from qgate.script import *
from qgate.script.qelib1 import *

if hasattr(qgate.simulator, 'cudaext') :

    class TestMultiDeviceCUDA(SimulatorTestBase) :

        def setUp(self) :
            #self.n_qregs = 3
            #self.n_lanes_in_chunk = 2
            self.n_qregs = 20
            self.n_lanes_in_chunk = 18
        
        def run_sim(self, circuit, multiDevice) :
            import qgate.qasm.script as script
            program = script.current_program()
            program = qgate.model.process(program, isolate_circuits=False)
            sim = qgate.simulator.cuda(program)

            n_lanes_in_chunk = -1
            device_ids = []
            if multiDevice :
                n_lanes_in_chunk = self.n_lanes_in_chunk
                device_ids = [0] * (1 << (self.n_qregs - self.n_lanes_in_chunk))
            
            sim.prepare(n_lanes_in_chunk, device_ids)
            sim.run()
            return sim

        def compare(self) :
            mc_states = self.run_sim(circuitTrue).get_qubits().get_states()
            sc_states = self.run_sim(circuitFalse).get_qubits().get_states()

            n_states = 1 << self.n_qregs
            self.assertTrue(np.allclose(mc_states, sc_states))

        def test_hadmard_gate(self) :
            circuit = new_circuit()
            qregs = allocate_qregs(self.n_qregs)
            circuit.add(h(qregs))
            self.compare()

        def test_cx_gate(self) :
            circuit = new_circuit()
            qregs = allocate_qregs(self.n_qregs)
            circuit.add(x(qregs[0]))
            for idx in range(0, self.n_qregs - 1) :
                circuit.add(cx(qregs[idx], qregs[idx + 1]))
            self.compare()

        def test_measure_x_mimimal(self) :
            circuit = new_circuit()
            qregs = allocate_qregs(self.n_qregs)
            init_cregs = allocate_cregs(self.n_qregs)
            neg_cregs = allocate_cregs(self.n_qregs)

            circuit.add(x(qregs[0]))
            circuit.add(measure(qregs[1], neg_cregs[1]))
            circuit.add(measure(qregs[2], neg_cregs[2]))

            sim = self.run_sim(circuitTrue)
            creg_values = sim.creg_values()
            
            self.assertEqual(creg_values.get(neg_cregs[1]), 0)

        def test_measure_x_minimal_2(self) :
            circuit = new_circuit()
            qregs = allocate_qregs(self.n_qregs)
            init_cregs = allocate_cregs(self.n_qregs)
            neg_cregs = allocate_cregs(1)

            for idx in range(0, self.n_qregs) :
                circuit.add(measure(qregs[idx], init_cregs[idx]))
            circuit.add(x(qregs[0]))
            circuit.add(measure(qregs[1], neg_cregs[0]))
            sim = self.run_sim(circuitTrue)
            creg_values = sim.creg_values()
            self.assertEqual(0, creg_values.get(neg_cregs[0]))

        def test_measure_x_minimal_3(self) :
            circuit = new_circuit()
            qregs = allocate_qregs(self.n_qregs)
            neg_cregs = allocate_cregs(self.n_qregs)

            circuit.add(x(qregs[2]))
            circuit.add(measure(qregs[0], neg_cregs[0]))
            circuit.add(measure(qregs[1], neg_cregs[1]))
            circuit.add(measure(qregs[2], neg_cregs[2]))
            sim = self.run_sim(circuitTrue)
            creg_values = sim.creg_values()
            
            self.assertEqual(0, creg_values.get(neg_cregs[0]))
            self.assertEqual(0, creg_values.get(neg_cregs[1]))
            self.assertEqual(1, creg_values.get(neg_cregs[2]))

        def test_measure_x(self) :
            for lane in range(0, self.n_qregs) :
                circuit = new_circuit()
                qregs = allocate_qregs(self.n_qregs)
                init_cregs = allocate_cregs(self.n_qregs)
                neg_cregs = allocate_cregs(self.n_qregs)

                for idx in range(0, self.n_qregs) :
                    circuit.add(measure(qregs[idx], init_cregs[idx]))
                circuit.add(x(qregs[lane]))
                for idx in range(0, self.n_qregs) :
                    circuit.add(measure(qregs[idx], neg_cregs[idx]))
                sim = self.run_sim(circuitTrue)
                creg_values = sim.creg_values()
                for idx in range(0, self.n_qregs) :
                    self.assertEqual(0, creg_values.get(init_cregs[idx]))
                    if idx == lane :
                        self.assertEqual(1, creg_values.get(neg_cregs[idx]))
                    else :
                        # print('lane {}, idx []'.format(lane, idx))
                        self.assertEqual(0, creg_values.get(neg_cregs[idx]))

                
if __name__ == '__main__':
    unittest.main()