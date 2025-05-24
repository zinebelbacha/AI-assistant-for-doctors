from src.models.densenet import XRayDenseNet
from src.utils.config import load_config
from src.utils.logging import setup_logging
import os

def main():
    logger = setup_logging()
    config = load_config()
    data_config = config['data']
    model_config = config['model']

    model = XRayDenseNet(
        input_shape=tuple(model_config['input_shape']),
        num_classes=model_config['num_classes'],
        model_path=os.path.join(model_config['model_dir'], 'final_densenet_model.keras')
    )
    loss, accuracy = model.evaluate(
        test_dir=os.path.join(data_config['extraction_dir'], 'test'),
        batch_size=model_config['batch_size']
    )
    logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()