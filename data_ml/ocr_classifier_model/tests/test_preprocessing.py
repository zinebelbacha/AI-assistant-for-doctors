import unittest
import cv2
import numpy as np
from src.data.preprocessing import preprocess_image, enhance_image

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test.png", self.test_image)

    def test_preprocess_image(self):
        result = preprocess_image("test.png")
        self.assertEqual(result.shape, (224, 224))

    def test_enhance_image(self):
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        result = enhance_image(gray_image)
        self.assertEqual(result.shape, gray_image.shape)

    def tearDown(self):
        os.remove("test.png")

if __name__ == '__main__':
    unittest.main()