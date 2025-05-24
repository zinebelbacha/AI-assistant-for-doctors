from src.data.dataset import XRayDataset
from src.models.densenet import XRayDenseNet
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    logger.info("Starting X-ray classifier pipeline")

    config = load_config()
    data_config = config['data']
    model_config = config['model']

    # Extract and organize dataset
    dataset = XRayDataset(
        zip_file_path=data_config['zip_file_path'],
        tb_zip_file_path=data_config['tb_zip_file_path'],
        extraction_dir=data_config['extraction_dir'],
        normal_dir=data_config['normal_dir']
    )
    dataset.extract_zip()
    dataset.move_normal_images(
        source_dir=os.path.join(data_config['extraction_dir'], 'TB_Chest_Radiography_Database/Normal')
    )
    dataset.organize_dataset(
        train_ratio=data_config['train_ratio'],
        test_ratio=data_config['test_ratio']
    )

    # Train model
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

    # Evaluate model
    loss, accuracy = model.evaluate(
        test_dir=os.path.join(data_config['extraction_dir'], 'test'),
        batch_size=model_config['batch_size']
    )
    logger.info(f"Final Test Loss: {loss:.4f}, Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()