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
        circuit = new_circuit()
        qreg0, qreg1 = new_qregs(2)
        circuit.add(a(qreg0),
                    h(qreg0),
                    s(qreg0),
                    t(qreg0),
                    x(qreg0),
                    y(qreg0),
                    z(qreg0),
                    rx(0.)(qreg0),
                    ry(0.)(qreg0),
                    rz(0.)(qreg0),
                    u1(0.)(qreg0),
                    u2(0., 0.)(qreg0),
                    u3(0., 0., 0.)(qreg0),
                    expia(0.)(qreg0),
                    expiz(0.)(qreg0),
                    sh(qreg0),
                    swap(qreg0, qreg1),
                    barrier(qreg0, qreg1))
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()
    def test_simple_ctrl_calls(self) :
        circuit = new_circuit()
        qreg0, qreg1 = new_qregs(2)
        circuit.add(cntr(qreg0).a(qreg1),
                    cntr(qreg0).h(qreg1),
                    cntr(qreg0).s(qreg1),
                    cntr(qreg0).t(qreg1),
                    cntr(qreg0).x(qreg1),
                    cntr(qreg0).y(qreg1),
                    cntr(qreg0).z(qreg1),
                    cntr(qreg0).rx(0.)(qreg1),
                    cntr(qreg0).ry(0.)(qreg1),
                    cntr(qreg0).rz(0.)(qreg1),
                    cntr(qreg0).u1(0.)(qreg1),
                    cntr(qreg0).u2(0., 0.)(qreg1),
                    cntr(qreg0).u3(0., 0., 0.)(qreg1),
                    cntr(qreg0).expia(0.)(qreg1),
                    cntr(qreg0).expiz(0.)(qreg1),
                    cntr(qreg0).sh(qreg1),
                    #swap(qreg1, qreg1),
        )
        self.run_sim(circuit)
        #try :
        #except :
        #    self.fail()

import sys
this = sys.modules[__name__]
createTestCases(this, 'TestSimpleCalls', TestSimpleCallsBase)
                
if __name__ == '__main__':
    unittest.main()
