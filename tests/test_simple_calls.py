from __future__ import print_function
from __future__ import absolute_import

from tests.test_base import *
from qgate.script import *
#from qgate.script.qelib1 import *

class TestSimpleCallsBase(SimulatorTestBase) :

    @classmethod
    def setUpClass(cls):
        if cls is TestSimpleCallsBase:
            raise unittest.SkipTest()
        super(TestSimpleCallsBase, cls).setUpClass()
        
    def run_sim(self, circuit) :
        sim = self._run_sim(circuit)
        return sim.qubits.get_states(qgate.simulator.prob)
    
    def test_simple_calls(self) :
        qreg0, qreg1 = new_qregs(2)
        circuit = [ A(qreg0),
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
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()
    def test_simple_ctrl_calls(self) :
        qreg0, qreg1 = new_qregs(2)
        circuit = [ ctrl(qreg0).A(qreg1),
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
                    ctrl(qreg0).SH(qreg1),
                    #swap(qreg1, qreg1),
        ]
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestSimpleCalls', TestSimpleCallsBase)
                
if __name__ == '__main__':
    unittest.main()
