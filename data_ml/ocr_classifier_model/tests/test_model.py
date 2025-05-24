import unittest
from src.models.cnn_model import SputumCNN

class TestSputumCNN(unittest.TestCase):
    def test_model_creation(self):
        cnn = SputumCNN(input_shape=(224, 224, 3), num_classes=3)
        self.assertIsNotNone(cnn.model)

if __name__ == '__main__':
    unittest.main()