import unittest
from src.data.generator import SputumReportGenerator
import os

class TestSputumReportGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_data"
        self.generator = SputumReportGenerator(self.output_dir)

    def test_generate_report(self):
        path = self.generator.generate_report(0, "train", "normal")
        self.assertTrue(os.path.exists(path))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()