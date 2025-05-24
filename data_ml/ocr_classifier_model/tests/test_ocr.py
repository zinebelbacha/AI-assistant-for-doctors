import unittest
from src.models.ocr import MedicalReportOCR
import cv2
import numpy as np

class TestMedicalReportOCR(unittest.TestCase):
    def setUp(self):
        self.ocr = MedicalReportOCR()
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite("test.png", self.test_image)

    def test_extract_text(self):
        result = self.ocr.extract_text("test.png")
        self.assertIn('full_text', result)

    def tearDown(self):
        os.remove("test.png")

if __name__ == '__main__':
    unittest.main()