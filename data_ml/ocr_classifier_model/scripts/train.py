from src.models.cnn_model import SputumCNN
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    logger.info("Starting model training")

    config = load_config()
    data_config = config['data']
    model_config = config['model']

    cnn = SputumCNN(
        input_shape=tuple(model_config['input_shape']),
        num_classes=model_config['num_classes']
    )
    cnn.train(
        train_dir=f"{data_config['output_dir']}/train",
        validation_dir=f"{data_config['output_dir']}/val",
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size']
    )
    logger.info("Training completed")

if __name__ == "__main__":
    main()