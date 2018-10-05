import unittest
import qgate

class SimulatorTestBase(unittest.TestCase) :
    
    def _run_sim(self) :
        import qgate.qasm.script as script
        program = script.current_program()
        program = qgate.qasm.process(program, isolate_circuits=False)
        sim = self.create_simulator(program)
        sim.prepare()
        sim.run()
        return sim

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
