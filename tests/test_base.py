import unittest
import qgate
import qgate.script as script

class SimulatorTestBase(unittest.TestCase) :
    
    def _run_sim(self, circuit, isolate_circuits = True) :
        circuit = script.process(circuit, isolate_circuits=isolate_circuits)
        sim = self.create_simulator(circuit)
        sim.prepare()
        sim.run()
        return sim

    def assertAlmostEqual(self, expected, actual) :
        unittest.TestCase.assertAlmostEqual(self, expected, actual, places = 5)



def create_py_simulator(self, program) :
    return qgate.simulator.py(program)

def create_cpu_simulator(self, program) :
    return qgate.simulator.cpu(program)

def create_cuda_simulator(self, program) :
    return qgate.simulator.cuda(program)


def createTestCases(module, class_name, base_class) :
    createPyTestCase(module, class_name, base_class)
    createCPUTestCase(module, class_name, base_class)
    createCUDATestCase(module, class_name, base_class)

def createPyTestCase(module, class_name, base_class) :
    py_class_name = class_name + 'Py'
    pytest_type = type(py_class_name, (base_class, ),
                       {"create_simulator":create_py_simulator})
    setattr(module, py_class_name, pytest_type)

def createCPUTestCase(module, class_name, base_class) :
    cpu_class_name = class_name + 'CPU'
    cputest_type = type(cpu_class_name, (base_class, ),
                        {"create_simulator":create_cpu_simulator})
    setattr(module, cpu_class_name, cputest_type)

def createCUDATestCase(module, class_name, base_class) :
    if hasattr(qgate.simulator, 'cudaruntime') :
        cuda_class_name = class_name + 'CUDA'
        cudatest_type = type(cuda_class_name, (base_class, ),
                            {"create_simulator":create_cuda_simulator})
        setattr(module, cuda_class_name, cudatest_type)
        
