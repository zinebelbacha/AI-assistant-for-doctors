import unittest
from src.models.densenet import XRayDenseNet

class TestXRayDenseNet(unittest.TestCase):
    def test_model_creation(self):
        model = XRayDenseNet(input_shape=(224, 224, 3), num_classes=3)
        self.assertIsNotNone(model.model)

if __name__ == '__main__':
    unittest.main()