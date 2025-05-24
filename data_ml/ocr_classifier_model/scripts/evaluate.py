from src.models.cnn_model import SputumCNN
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    logger.info("Starting model evaluation")

    config = load_config()
    data_config = config['data']
    model_config = config['model']

    cnn = SputumCNN(
        input_shape=tuple(model_config['input_shape']),
        num_classes=model_config['num_classes']
    )
    loss, accuracy = cnn.evaluate(f"{data_config['output_dir']}/test")
    logger.info(f"Test Loss: {loss}, Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()