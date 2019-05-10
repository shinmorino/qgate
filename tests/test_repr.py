from __future__ import print_function
from __future__ import absolute_import

import unittest
import qgate
from qgate.script import *

class TestRepr(unittest.TestCase) :

    def setUp(self) :
        import io
        self.file = io.StringIO()

    def tearDown(self) :
        # print(self.file.getvalue())
        pass
    
    def test_dump_circuit(self) :
        qreg0, qreg1 = new_qregs(2)
        
        circuit = [ I(qreg0),
                    H(qreg0),
                    S(qreg0),
                    T(qreg0),
                    X(qreg0),
                    Y(qreg0),
                    Z(qreg0),
                    Rx(0.)(qreg0),
                    Ry(0.)(qreg0),
                    Rz(0.)(qreg0),
                    U1(0.)(qreg0),
                    U2(0., 0.)(qreg0),
                    U3(0., 0., 0.)(qreg0),
                    Expia(0.)(qreg0),
                    Expiz(0.)(qreg0),
                    SH(qreg0),
                    Swap(qreg0, qreg1),
                    barrier(qreg0, qreg1) ]
        
        circuit += [ I.Adj(qreg0),
                     H.Adj(qreg0),
                     S.Adj(qreg0),
                     T.Adj(qreg0),
                     X.Adj(qreg0),
                     Y.Adj(qreg0),
                     Z.Adj(qreg0),
                     Rx(0.).Adj(qreg0),
                     Ry(0.).Adj(qreg0),
                     Rz(0.).Adj(qreg0),
                     U1(0.).Adj(qreg0),
                     U2(0., 0.).Adj(qreg0),
                     U3(0., 0., 0.).Adj(qreg0),
                     Expia(0.).Adj(qreg0),
                     Expiz(0.).Adj(qreg0),
                     SH.Adj(qreg0),
                     Swap(qreg0, qreg1),
                    barrier(qreg0, qreg1) ]
        
        circuit += [ ctrl(qreg0).I(qreg1),
                     ctrl(qreg0).H(qreg1),
                     ctrl(qreg0).S(qreg1),
                     ctrl(qreg0).T(qreg1),
                     ctrl(qreg0).X(qreg1),
                     ctrl(qreg0).Y(qreg1),
                     ctrl(qreg0).Z(qreg1),
                     ctrl(qreg0).Rx(0.)(qreg1),
                     ctrl(qreg0).Ry(0.)(qreg1),
                     ctrl(qreg0).Rz(0.)(qreg1),
                     ctrl(qreg0).U1(0.)(qreg1),
                     ctrl(qreg0).U2(0., 0.)(qreg1),
                     ctrl(qreg0).U3(0., 0., 0.)(qreg1),
                     ctrl(qreg0).Expia(0.)(qreg1),
                     ctrl(qreg0).Expiz(0.)(qreg1),
                     ctrl(qreg0).SH(qreg1)]
        
        circuit += [ ctrl(qreg0).I.Adj(qreg1),
                     ctrl(qreg0).H.Adj(qreg1),
                     ctrl(qreg0).S.Adj(qreg1),
                     ctrl(qreg0).T.Adj(qreg1),
                     ctrl(qreg0).X.Adj(qreg1),
                     ctrl(qreg0).Y.Adj(qreg1),
                     ctrl(qreg0).Z.Adj(qreg1),
                     ctrl(qreg0).Rx(0.).Adj(qreg1),
                     ctrl(qreg0).Ry(0.).Adj(qreg1),
                     ctrl(qreg0).Rz(0.).Adj(qreg1),
                     ctrl(qreg0).U1(0.).Adj(qreg1),
                     ctrl(qreg0).U2(0., 0.).Adj(qreg1),
                     ctrl(qreg0).U3(0., 0., 0.).Adj(qreg1),
                     ctrl(qreg0).Expia(0.).Adj(qreg1),
                     ctrl(qreg0).Expiz(0.).Adj(qreg1),
                     ctrl(qreg0).SH(qreg1)]
        
        ref = new_reference()
        circuit += [measure(ref, qreg0)]
        qgate.dump(circuit, file = self.file)

    def test_dump_qubits(self) :
        qregs = new_qregs(4)
        circuit = [H(qreg) for qreg in qregs]
        sim = qgate.simulator.py()
        sim.run(circuit)
        qgate.dump(sim.qubits.states, file = self.file)
        qgate.dump(sim.qubits.prob, file = self.file)

    def test_dump_values(self) :
        qregs = new_qregs(4)
        cregs = new_references(4)
        circuit = [I(qreg) for qreg in qregs]
        sim = qgate.simulator.py()
        sim.run(circuit)
        qgate.dump(sim.values, file = self.file)


if __name__ == '__main__':
    unittest.main()
