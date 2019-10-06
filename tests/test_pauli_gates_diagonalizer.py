from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.simulator.pyruntime import adjoint
from qgate.model.pauli_gates_diagonalizer import PauliGatesDiagonalizer
from qgate.model import gate_type as gtype
from qgate.model.expand import expand_exp
import numpy as np

class TestPauliGatesDiagonalizer(SimulatorTestBase) :

    def diagonalize(self, paulis) :
        diag = PauliGatesDiagonalizer(paulis)
        is_z_based = diag.diagonalize()
        pcx = diag.get_pcx()
        phase_offset = diag.get_phase_offset_in_pi_2()
        return is_z_based, pcx, phase_offset

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

    def test_I(self) :
        qreg = new_qreg()
        gates = [ I(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)

    def test_Ix3(self) :
        qregs = new_qregs(3)
        gates = [ I(qregs[0]), I(qregs[1]), I(qregs[2]) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)

    def test_Z(self) :
        qreg = new_qreg()
        gates = [ Z(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, True)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)

    def test_Zx3(self) :
        qregs = new_qregs(3)
        gates = [ Z(qregs[0]), Z(qregs[1]), Z(qregs[2]) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, True)
        self.assertEqual(len(pcx), 2)
        self.assertTrue(all([isinstance(gate.gate_type, gtype.X) for gate in pcx]))
        x0, x1 = pcx
        self.assertEqual(x0.qreg, qregs[1])
        self.assertEqual(x0.ctrllist, [qregs[0]])
        self.assertEqual(x1.qreg, qregs[2])
        self.assertEqual(x1.ctrllist, [qregs[1]])
        
        self.assertEqual(phase_offset, 0)

    def test_X(self) :
        qreg = new_qreg()
        gates = [ X(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, True)
        self.assertEqual(len(pcx), 1)
        self.assertTrue(isinstance(pcx[0].gate_type, gtype.H))
        self.assertEqual(phase_offset, 0)

    def test_Xx3(self) :
        qregs = new_qregs(3)
        gates = [ X(qregs[0]), X(qregs[1]), X(qregs[2]) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, True)
        self.assertEqual(len(pcx), 5)
        self.assertTrue(all([isinstance(gate.gate_type, gtype.H) for gate in pcx[:3]]))
        h0, h1, h2 = pcx[:3]
        self.assertEqual(h0.qreg, qregs[0])
        self.assertEqual(h1.qreg, qregs[1])
        self.assertEqual(h2.qreg, qregs[2])
        
        self.assertTrue(all([isinstance(gate.gate_type, gtype.X) for gate in pcx[3:]]))
        x0, x1 = pcx[3:]
        self.assertEqual(x0.qreg, qregs[1])
        self.assertEqual(x0.ctrllist, [qregs[0]])
        self.assertEqual(x1.qreg, qregs[2])
        self.assertEqual(x1.ctrllist, [qregs[1]])

        self.assertEqual(phase_offset, 0)

    def test_Y(self) :
        qreg = new_qreg()
        gates = [ Y(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, True)
        self.assertEqual(len(pcx), 1)
        self.assertTrue(isinstance(pcx[0].gate_type, gtype.SH))
        self.assertEqual(phase_offset, 0)

    def test_Yx3(self) :
        qregs = new_qregs(3)
        gates = [ Y(qregs[0]), Y(qregs[1]), Y(qregs[2]) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, True)
        self.assertEqual(len(pcx), 5)
        self.assertTrue(all([isinstance(gate.gate_type, gtype.SH) for gate in pcx[:3]]))
        sh0, sh1, sh2 = pcx[:3]
        self.assertEqual(sh0.qreg, qregs[0])
        self.assertEqual(sh1.qreg, qregs[1])
        self.assertEqual(sh2.qreg, qregs[2])

        self.assertTrue(all([isinstance(gate.gate_type, gtype.X) for gate in pcx[3:]]))
        x0, x1 = pcx[3:]
        self.assertEqual(x0.qreg, qregs[1])
        self.assertEqual(x0.ctrllist, [qregs[0]])
        self.assertEqual(x1.qreg, qregs[2])
        self.assertEqual(x1.ctrllist, [qregs[1]])

        self.assertEqual(phase_offset, 0)

    def test_xyz_simplification(self) :
        qreg = new_qreg()
        xyz = [ X(qreg), Y(qreg), Z(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(xyz)

        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 1)

        gates = xyz + xyz
        is_z_based, pcx, phase_offset = self.diagonalize(gates)
        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 2)

        gates = xyz + xyz + xyz
        is_z_based, pcx, phase_offset = self.diagonalize(gates)
        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 3)

        gates = xyz + xyz + xyz + xyz
        is_z_based, pcx, phase_offset = self.diagonalize(gates)
        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)

    def test_simplification_same_paulis(self) :
        qreg = new_qreg()
        gates = [ X(qreg), X(qreg), Y(qreg), Y(qreg), Z(qreg), Z(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)

        gates = [ X(qreg), Y(qreg), Z(qreg), Z(qreg), Y(qreg), X(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)
        
        gates = [ X(qreg), Y(qreg), Z(qreg), Z(qreg), X(qreg), X(qreg), Y(qreg), X(qreg) ]
        is_z_based, pcx, phase_offset = self.diagonalize(gates)

        self.assertEqual(is_z_based, False)
        self.assertEqual(len(pcx), 0)
        self.assertEqual(phase_offset, 0)

if __name__ == '__main__':
    unittest.main()
