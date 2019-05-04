
files = [
    'examples/ibmqx2/iswap.qasm',
    'examples/ibmqx2/W3test.qasm',
    'examples/ibmqx2/qe_qft_3.qasm',
    'examples/ibmqx2/011_3_qubit_grover_50_.qasm',
    'examples/ibmqx2/Deutsch_Algorithm.qasm',
    'examples/ibmqx2/qe_qft_5.qasm',
    'examples/ibmqx2/qe_qft_4.qasm',
    # 'examples/generic/teleportv2.qasm',
    # 'examples/generic/teleport.qasm',
    'examples/generic/inverseqft1.qasm',
    'examples/generic/inverseqft2.qasm',
    'examples/generic/qft.qasm',
    #'examples/generic/W-state.qasm',
    #'examples/generic/ipea_3_pi_8.qasm',
    #'examples/generic/pea_3_pi_8.qasm',
    #'examples/generic/qpt.qasm',
    #'examples/generic/qec.qasm',
    #'examples/generic/adder.qasm',
    #'examples/generic/rb.qasm',
    #'examples/generic/bigadder.qasm',
    'benchmarks/qft/qft_n13.qasm',
    'benchmarks/qft/qft_n10.qasm',
    'benchmarks/qft/qft_n14.qasm',
    'benchmarks/qft/qft_n12.qasm',
    'benchmarks/qft/qft_n16.qasm',
    'benchmarks/qft/qft_n11.qasm',
    'benchmarks/qft/qft_n18.qasm',
    'benchmarks/qft/qft_n19.qasm',
    'benchmarks/qft/qft_n20.qasm',
    'benchmarks/qft/qft_n15.qasm',
    'benchmarks/qft/qft_n17.qasm',
    'benchmarks/cc/cc_n13.qasm',
    'benchmarks/cc/cc_n15.qasm',
    'benchmarks/cc/cc_n12.qasm',
    'benchmarks/cc/cc_n17.qasm',
    'benchmarks/cc/cc_n16.qasm',
    'benchmarks/cc/cc_n10.qasm',
    'benchmarks/cc/cc_n19.qasm',
    'benchmarks/cc/cc_n18.qasm',
    'benchmarks/cc/cc_n14.qasm',
    'benchmarks/cc/cc_n11.qasm',
    'benchmarks/sat/sat_n7_vars=3_clauses=3_clauselen=3_fd2ff8bf99b05c13216600154bc6bc21.qasm',
    'benchmarks/sat/sat_n8_vars=2_clauses=4_clauselen=2_c0e5b4bc4a7708567943d6880c194173.qasm',
    'benchmarks/sat/sat_n9_vars=4_clauses=3_clauselen=4_f08ded5404079f6332615c302267c9d9.qasm',
    'benchmarks/sat/sat_n6_vars=2_clauses=3_clauselen=2_52463030d15e7244c9f75bf819d40e3d.qasm',
    'benchmarks/sat/sat_n10_vars=4_clauses=4_clauselen=4_fb92209893909945c2bc1144e6972464.qasm',
    'benchmarks/bv/bv_n10.qasm',
    'benchmarks/bv/bv_n16.qasm',
    'benchmarks/bv/bv_n15.qasm',
    'benchmarks/bv/bv_n13.qasm',
    'benchmarks/bv/bv_n19.qasm',
    'benchmarks/bv/bv_n11.qasm',
    'benchmarks/bv/bv_n17.qasm',
    'benchmarks/bv/bv_n12.qasm',
    'benchmarks/bv/bv_n14.qasm',
    'benchmarks/bv/bv_n18.qasm',
    'benchmarks/quantum_volume/quantum_volume_n40_d32.qasm',
    'benchmarks/quantum_volume/quantum_volume_n5_d4.qasm',
    'benchmarks/quantum_volume/quantum_volume_n5_d5.qasm',
    'benchmarks/quantum_volume/quantum_volume_n34_d8.qasm',
    'benchmarks/quantum_volume/quantum_volume_n38_d38.qasm',
    'benchmarks/quantum_volume/quantum_volume_n5_d3.qasm',
    'benchmarks/quantum_volume/quantum_volume_n35_d8.qasm',
    'benchmarks/quantum_volume/quantum_volume_n5_d2.qasm',
    'benchmarks/quantum_volume/quantum_volume_n36_d8.qasm',
    'benchmarks/quantum_volume/quantum_volume_n40_d40.qasm',
    'benchmarks/quantum_volume/quantum_volume_n32_d32.qasm',
]

invalid_files = [
    'examples/invalid/gate_no_found.qasm',
    'examples/invalid/missing_semicolon.qasm'
]

import qgate
import unittest
import os

# prefix for openqasm git repo dir.
prefix = '.'
openqasm_dir_found = False
while True :
    if os.path.isdir(prefix) :
        if os.path.isdir(prefix + '/openqasm') :
            openqasm_dir_found = True
            break
        else :
            prefix += '/..'
    else :
        break

openqasm_dir = prefix + '/openqasm'

class TestLoadOpenQASM(unittest.TestCase) :
    @unittest.skipIf(not openqasm_dir_found, 'openqasm dir not found')
    def test_load(self) :
        for filename in files :
            fullpath = openqasm_dir + '/' + filename

            # generate source module
            qasm = qgate.openqasm.translate_file(fullpath)
            # print(qasm)
            mod = qgate.openqasm.load_circuit_from_file(fullpath)
            #sim = qgate.simulator.cpu()
            #sim.run(mod.circuit)
            
    @unittest.skipIf(not openqasm_dir_found, 'openqasm dir not found')
    def test_load_fail(self) :
        for filename in invalid_files :
            fullpath = openqasm_dir + '/' + filename
            # generate source module
            with self.assertRaises(Exception):
                qasm = qgate.openqasm.translate_file(fullpath)

if __name__ == '__main__':
    unittest.main()
