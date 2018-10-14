import qgate
import unittest

class TestCUDAEnabled(unittest.TestCase) :
    def test_import_cuda(self) :
        self.assertTrue(hasattr(qgate.simulator, 'cudaruntime'))

if __name__ == '__main__':
    unittest.main()
