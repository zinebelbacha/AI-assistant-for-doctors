from src.models.densenet import XRayDenseNet
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    config = load_config()
    data_config = config['data']
    model_config = config['model']

    model = XRayDenseNet(
        input_shape=tuple(model_config['input_shape']),
        num_classes=model_config['num_classes']
    )
    model.train(
        train_dir=os.path.join(data_config['extraction_dir'], 'train'),
        validation_dir=os.path.join(data_config['extraction_dir'], 'val'),
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size']
    )
    logger.info("Training completed")

if __name__ == "__main__":
    main()