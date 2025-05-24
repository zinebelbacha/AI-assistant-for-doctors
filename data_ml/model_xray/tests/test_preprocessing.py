import unittest
import cv2
import numpy as np
from src.data.preprocessing import preprocess_image

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test.png", self.test_image)

    def test_preprocess_image(self):
        result = preprocess_image("test.png")
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertTrue(result.max() <= 1.0)

    def tearDown(self):
        os.remove("test.png")

if __name__ == '__main__':
    unittest.main()