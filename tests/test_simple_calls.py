from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
from qgate.script.qelib1 import *
#from qgate.script.qelib1 import *

class TestSimpleCallsBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestSimpleCallsBase:
            raise unittest.SkipTest()
        super(TestSimpleCallsBase, cls).setUpClass()
        
    def run_sim(self, circuit) :
        # test dump
        import os
        with open(os.devnull,"w") as devnull :
            qgate.model.dump(circuit, file = devnull)

        sim = self._run_sim(circuit)
        return sim.qubits.get_states(qgate.simulator.prob)
    
    def test_simple_calls(self) :
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
                    Expii(0.)(qreg0),
                    Expiz(0.)(qreg0),
                    SH(qreg0),
                    Swap(qreg0, qreg1),
                    barrier(qreg0, qreg1) ]
        self.run_sim(circuit)

    def test_simple_adj_calls(self) :
        qreg0, qreg1 = new_qregs(2)
        circuit = [ I.Adj(qreg0),
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
                    Expii(0.).Adj(qreg0),
                    Expiz(0.).Adj(qreg0),
                    SH.Adj(qreg0),
                    Swap(qreg0, qreg1),
                    barrier(qreg0, qreg1) ]
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()

    def test_simple_ctrl_calls(self) :
        qreg0, qreg1 = new_qregs(2)
        circuit = [ ctrl(qreg0).I(qreg1),
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
                    ctrl(qreg0).Expii(0.)(qreg1),
                    ctrl(qreg0).Expiz(0.)(qreg1),
                    ctrl(qreg0).SH(qreg1),
                    #swap(qreg1, qreg1),
        ]
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()

    def test_simple_ctrl_adj_calls(self) :
        qreg0, qreg1 = new_qregs(2)
        circuit = [ ctrl(qreg0).I.Adj(qreg1),
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
                    ctrl(qreg0).Expii(0.).Adj(qreg1),
                    ctrl(qreg0).Expiz(0.).Adj(qreg1),
                    ctrl(qreg0).SH(qreg1),
                    #swap(qreg1, qreg1),
        ]
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()

    def test_qelib1_calls(self) :
        qreg0, qreg1, qreg2 = new_qregs(3)
        circuit = [ cx(qreg0, qreg1),
                    sdg(qreg0),
                    tdg(qreg0),
                    cz(qreg0, qreg1),
                    cy(qreg0, qreg1),
                    ch(qreg0, qreg1),
                    ccx(qreg0, qreg1, qreg2),
                    crz(0., qreg0, qreg1),
                    cu1(0., qreg0, qreg1),
                    cu3(0., 0., 0., qreg0, qreg1),
        ]
        self.run_sim(circuit)

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestSimpleCalls', TestSimpleCallsBase)
                
if __name__ == '__main__':
    unittest.main()
