import unittest
import qgate
import numpy as np

if hasattr(qgate.simulator, 'cudaruntime') :

    class TestMemstore(unittest.TestCase) :

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

        def test_memstore(self) :
            qgate.simulator.cudaruntime.module_finalize()

            # allocate 4 chunks whose sizes are  max_po2idx_per_chunk.
            po2idx_per_chunk = self.n_qregs + 3 - 2  # 3 is for po2idx of sizeof(float complex).
            memstore_size = 1 << (self.n_qregs + 3)
            
            qgate.simulator.cudaruntime.set_preference(
                device_ids=[0], max_po2idx_per_chunk= po2idx_per_chunk, memory_store_size=memstore_size)

            qstates = qgate.simulator.cudaruntime.create_qubit_states(np.float32)
            # internally allocate 4 chunks
            qstates.processor.initialize_qubit_states(qstates, self.n_qregs)
            # delete internal buffer
            qstates.delete()

            qstates = qgate.simulator.cudaruntime.create_qubit_states(np.float32)
            # purging cache, and reallocate a smaller chunk.
            # One of cached chunks is released to make free memory.
            qstates.processor.initialize_qubit_states(qstates, self.n_qregs - 2)
            # delete internal buffer
            qstates.delete()

            qstates = qgate.simulator.cudaruntime.create_qubit_states(np.float32)
            # internally allocate 4 chunks again.
            # The cached chunk is internally released.
            qstates.processor.initialize_qubit_states(qstates, self.n_qregs)
            # delete internal buffer
            qstates.delete()

            qgate.simulator.cudaruntime.module_finalize()

            self.assertTrue(True)

                
if __name__ == '__main__':
    unittest.main()
