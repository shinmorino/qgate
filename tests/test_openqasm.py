from __future__ import print_function
from __future__ import absolute_import
from qgate.openqasm import load_circuit, translate

import unittest

class TestOpenQASM(unittest.TestCase) :
    
    def test_load_empty_qasm(self) :
        qasm = ''
        mod = load_circuit(qasm)
    
    def test_decl(self) :
        qasm = 'qreg q[10]; creg c[10];'
        mod = load_circuit(qasm)
        
    def test_decl(self) :
        qasm = 'qreg q[10]; creg c[10];'
        mod = load_circuit(qasm)

    def test_gatedecl(self) :
        qasm = 'qreg q[10]; creg c[10];' + \
               'gate custome_gate() a, b, c { }'
        with self.assertRaises(RuntimeError):
            mod = load_circuit(qasm)

    def test_opaque(self) :
        qasm = 'qreg q[10]; creg c[10];\n' + \
               'opaque opaque_gate() a, b, c;'
        with self.assertRaises(NotImplementedError):
            mod = load_circuit(qasm)

    def test_U(self) :
        qasm = 'qreg q[10]; creg c[10];' + \
               'U(0., 0., 0.) q[2];' + \
               'U(0., 0., 0.) q;'
        mod = load_circuit(qasm)

    def test_U_arguments(self) :
        qasm = 'qreg q[10]; creg c[10];' + \
               'U(0., 0., 0.) q[0];' + \
               'U(0., 0., 0.) q;'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_CX(self) :
        qasm = 'qreg q[10];' + \
               'CX q[2], q[3];' 
        mod = load_circuit(qasm)

    def test_id_gate_anylist(self) :
        qasm = 'qreg q[10];' + \
               'h q[2], q[3];' 
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_id_gate_anylist_mixed(self) :
        qasm = 'qreg q[10];' + \
               'h q[2], q;' 
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_id_gate_with_explist(self) :
        # FIXME: check # params.
        qasm = 'qreg q[10];' + \
               'u1(0.) q[0];' + \
               'u2(0., 0.) q[0];' + \
               'u3(0., 0., 0.) q[0];' 
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_measure(self) :
        qasm = 'qreg q[10]; creg c[10];' + \
               'measure q[0] -> c[0];'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_measure_id(self) :
        qasm = 'qreg q[10]; creg c[10];' + \
               'measure q -> c;'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_reset(self) :
        qasm = 'qreg q[10];' + \
               'reset q[0];'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_reset_id(self) :
        qasm = 'qreg q[10];' + \
               'reset q;'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_barrier(self) :
        qasm = 'qreg q[10];' + \
               'barrier q[0];'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_barrier_id(self) :
        qasm = 'qreg q[10];' + \
               'barrier q;'
        # print(translate(qasm))
        mod = load_circuit(qasm)

    def test_exec_error(self) :
        qasm = 'qreg q[10];' + \
               'barrier g;'
        # print(translate(qasm))
        with self.assertRaises(NameError):
            mod = load_circuit(qasm)

    def test_lex_error(self) :
        qasm = 'qreg q[10];\n' + \
               'h q[0];\n' + \
               'h q[1];\n' + \
               '_h q[2];\n' + \
               'h q[3];\n' + \
               'h q[4];\n'
        # print(translate(qasm))
        with self.assertRaises(SyntaxError):
            mod = load_circuit(qasm)

    def test_yacc_error(self) :
        qasm = 'qreg q[10];\n' + \
               'h q[0]\n' + \
               'h q[1];\n' + \
               '_h q[2];\n' + \
               'h q[3];\n' + \
               'h q[4];\n'
        # print(translate(qasm))
        with self.assertRaises(SyntaxError):
            mod = load_circuit(qasm)
        
if __name__ == '__main__':
    unittest.main()
