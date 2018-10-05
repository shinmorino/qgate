import unittest
import qgate

class SimulatorTestBase(unittest.TestCase) :
    
    def _run_sim(self) :
        import qgate.qasm.script as script
        program = script.current_program()
        program = qgate.model.process(program, isolate_circuits=True)
        sim = self.create_simulator(program)
        sim.prepare()
        sim.run()
        return sim

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)
