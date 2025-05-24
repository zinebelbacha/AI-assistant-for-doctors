import unittest
import os
import shutil
from src.data.dataset import XRayDataset

class TestXRayDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        self.normal_dir = os.path.join(self.test_dir, "NORMAL")
        os.makedirs(self.normal_dir, exist_ok=True)
        with open(os.path.join(self.normal_dir, "test.png"), 'w') as f:
            f.write("dummy")
        self.dataset = XRayDataset(
            zip_file_path="dummy.zip",
            tb_zip_file_path="dummy_tb.zip",
            extraction_dir=self.test_dir,
            normal_dir=self.normal_dir
        )

    def test_count_files(self):
        count = self.dataset.count_files(self.normal_dir)
        self.assertEqual(count, 1)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()