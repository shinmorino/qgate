from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.simulator.pyruntime import adjoint
from qgate.model import gate_type as gtype
from qgate.model.expand import expand_exp, expand_pmeasure, expand_pprob
from qgate import model
import numpy as np
import cmath

class TestPauliMacroExpansion(SimulatorTestBase) :

    def get_matrix(self, gate) :
        mat = gate.gate_type.pymat()
        if gate.adjoint :
            mat = adjoint(mat)
        return mat

    def gate_matrix_product(self, ops, qregs) :
        matdim = 1 << len(qregs)
        product = np.eye(matdim, dtype=np.complex128)
        eyeNN = np.eye(matdim, dtype=np.complex128)
        for gate in ops :
            # creating permutation index table
            lane = qregs.index(gate.qreg)
            permuted = np.zeros((matdim, ), np.int)
            mask_0 = (np.int(1) << lane) - 1
            mask_1 = np.int(-1) & ~((np.int(2) << lane) - 1)
            lane_bit = np.int(1) << lane
            for idx in range(matdim // 2) :
                idx_0 = ((idx << 1) & mask_1) | (idx & mask_0)
                idx_1 = idx_0 | lane_bit
                permuted[idx * 2] = idx_0
                permuted[idx * 2 + 1] = idx_1
            indices = np.linspace(0, matdim - 1, matdim, dtype=np.int)
            # caluate matrix, mat22 is at lane 0.
            mat22 = self.get_matrix(gate)
            eyeNN_1 = np.eye(matdim // 2, dtype=np.complex128)
            matNN = np.kron(eyeNN_1, mat22)
            # permutation
            matNN[:, permuted] = matNN[:, indices]
            matNN[permuted, :] = matNN[indices,:]

            if gate.ctrllist :
                ctrl_bit = 1 << qregs.index(gate.ctrllist[0])
                for idx in range(matdim) :
                    if (ctrl_bit & idx) == 0:
                        matNN[:, idx] = eyeNN[:, idx]
                        matNN[idx, :] = eyeNN[idx, :]

            product = np.matmul(matNN, product)
                
        return product

    def test_expi(self) :
        qregs = new_qregs(3)
        paulis = [X(qregs[2]), Y(qregs[1]), Z(qregs[0])]
        exp = Expi(1)(paulis)
        expanded = expand_exp(exp)

        # test convert to z
        expgate, = [gate for gate in expanded if isinstance(gate.gate_type, gtype.ExpiZ)]
        expidx = expanded.index(expgate)
        pcx = expanded[:expidx]
        pcxadj = expanded[expidx+1:]
        product = self.gate_matrix_product(pcxadj + paulis + pcx, qregs)
        
        # create z gate product being measured.
        zlist = list()
        for qreg in qregs :
            gate = Z(qreg) if expgate.qreg == qreg else I(qreg)
            zlist.append(gate)
        # exp.gate_type.args[0] is exp(j * [phase offset]).
        d = exp.gate_type.args[0] * self.gate_matrix_product(zlist, qregs)
        self.assertTrue(np.allclose(product, d))

    def test_unexecutable_expi(self) :
        qreg = new_qreg()
        paulis = [X(qreg), Y(qreg)]
        exp = Expi(1)(paulis)
        with self.assertRaises(RuntimeError):
            expanded = expand_exp(exp)

    def test_pmeasure(self) :
        qregs = new_qregs(3)
        paulis = [X(qregs[2]), Y(qregs[2]), Y(qregs[1]), Z(qregs[0])]
        ref = new_reference()
        pmeasure = measure(ref, paulis)
        expanded = expand_pmeasure(pmeasure)
        # qgate.dump(expanded)

        # test convert to z
        mop, = [op for op in expanded if isinstance(op, model.Measure)]
        midx = expanded.index(mop)
        pcx = expanded[:midx]
        pcxadj = expanded[midx+1:]

        product = self.gate_matrix_product(pcxadj + pcx, qregs)
        self.assertTrue(np.allclose(product, np.eye(8, dtype=np.complex128)))

        product = self.gate_matrix_product(pcxadj + paulis + pcx, qregs)
        product /= product[0, 0]
        # print(product)
        
        # create z gate product being measured.
        zlist = list()
        for qreg in qregs :
            gate = Z(qreg) if mop.qreg == qreg else I(qreg)
            zlist.append(gate)
        z = self.gate_matrix_product(zlist, qregs)
        self.assertTrue(np.allclose(product, z))

    def test_pprob(self) :
        qregs = new_qregs(3)
        paulis = [X(qregs[2]), Y(qregs[2]), Y(qregs[1]), Z(qregs[0])]
        ref = new_reference()
        pprob = prob(ref, paulis)
        expanded = expand_pprob(pprob)
        # qgate.dump(expanded)

        # test convert to z
        pop, = [op for op in expanded if isinstance(op, model.Prob)]
        pidx = expanded.index(pop)
        pcx = expanded[:pidx]
        pcxadj = expanded[pidx+1:]

        product = self.gate_matrix_product(pcxadj + pcx, qregs)
        self.assertTrue(np.allclose(product, np.eye(8, dtype=np.complex128)))

        product = self.gate_matrix_product(pcxadj + paulis + pcx, qregs)
        product /= product[0, 0]
        # print(product)
        
        # create z gate product being measured.
        zlist = list()
        for qreg in qregs :
            gate = Z(qreg) if pop.qreg == qreg else I(qreg)
            zlist.append(gate)
        z = self.gate_matrix_product(zlist, qregs)
        self.assertTrue(np.allclose(product, z))

if __name__ == '__main__':
    unittest.main()
