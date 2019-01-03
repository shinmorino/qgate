from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append('C:\\projects\\qgate_sandbox')

from tests.test_base import *
from qgate.qasm.script import *
from qgate.qasm.qelib1 import *

if hasattr(qgate.simulator, 'cudaext') :

    class TestMultiDeviceCUDA(SimulatorTestBase) :

        def setUp(self) :
            self.n_qregs = 3
            self.n_lanes_in_chunk = 2
        
        def run_sim(self, multiDevice) :
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
            mc_states = self.run_sim(True).get_qubits().get_states()
            sc_states = self.run_sim(False).get_qubits().get_states()

            n_states = 1 << self.n_qregs
            self.assertTrue(np.allclose(mc_states, sc_states))

        def test_hadmard_gate(self) :
            new_program()
            qregs = allocate_qreg(self.n_qregs)
            op(h(qregs))
            self.compare()
            fin_program()

        def test_cx_gate(self) :
            new_program()
            qregs = allocate_qreg(self.n_qregs)
            op(x(qregs[0]))
            for idx in range(0, self.n_qregs - 1) :
                op(cx(qregs[idx], qregs[idx + 1]))
            self.compare()
            fin_program()

        def test_measure_x_mimimal(self) :
            new_program()
            qregs = allocate_qreg(self.n_qregs)
            init_cregs = allocate_creg(self.n_qregs)
            neg_cregs = allocate_creg(self.n_qregs)

            op(x(qregs[0]))
            op(measure(qregs[1], neg_cregs[1]))
            op(measure(qregs[2], neg_cregs[2]))

            sim = self.run_sim(True)
            creg_dict = sim.get_creg_dict()
            
            self.assertEqual(creg_dict.get_value(neg_cregs[1]), 0)
            fin_program()

        def test_measure_x_minimal_2(self) :
            new_program()
            qregs = allocate_qreg(self.n_qregs)
            init_cregs = allocate_creg(self.n_qregs)
            neg_cregs = allocate_creg(1)

            for idx in range(0, self.n_qregs) :
                op(measure(qregs[idx], init_cregs[idx]))
            op(x(qregs[0]))
            op(measure(qregs[1], neg_cregs[0]))
            sim = self.run_sim(True)
            creg_dict = sim.get_creg_dict()
            self.assertEqual(0, creg_dict.get_value(neg_cregs[0]))
            fin_program()

        def test_measure_x(self) :
            for lane in range(0, self.n_qregs) :
                new_program()
                qregs = allocate_qreg(self.n_qregs)
                init_cregs = allocate_creg(self.n_qregs)
                neg_cregs = allocate_creg(self.n_qregs)

                for idx in range(0, self.n_qregs) :
                    op(measure(qregs[idx], init_cregs[idx]))
                op(x(qregs[lane]))
                for idx in range(0, self.n_qregs) :
                    op(measure(qregs[idx], neg_cregs[idx]))
                sim = self.run_sim(True)
                creg_dict = sim.get_creg_dict()
                for idx in range(0, self.n_qregs) :
                    self.assertEqual(0, creg_dict.get_value(init_cregs[idx]))
                    if idx == lane :
                        self.assertEqual(1, creg_dict.get_value(neg_cregs[idx]))
                    else :
                        # print('lane {}, idx []'.format(lane, idx))
                        self.assertEqual(0, creg_dict.get_value(neg_cregs[idx]))
                fin_program()

                
if __name__ == '__main__':
    unittest.main()
