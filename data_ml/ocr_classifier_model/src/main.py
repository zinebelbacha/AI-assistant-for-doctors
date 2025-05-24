from src.data.generator import SputumReportGenerator
from src.models.cnn_model import SputumCNN
from src.models.ocr import MedicalReportOCR
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    logger.info("Starting sputum cytology OCR pipeline")

    config = load_config()
    data_config = config['data']
    model_config = config['model']

    # Generate data
    logger.info("Generating fake dataset")
    generator = SputumReportGenerator(
        output_dir=data_config['output_dir'],
        image_size=tuple(data_config['image_size']),
        font_path=data_config['font_path']
    )
    generator.generate_dataset(
        num_reports=data_config['num_reports'],
        train_ratio=data_config['train_ratio'],
        test_ratio=data_config['test_ratio']
    )

    # Train model
    logger.info("Training CNN model")
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

    # Evaluate model
    logger.info("Evaluating model")
    loss, accuracy = cnn.evaluate(f"{data_config['output_dir']}/test")
    logger.info(f"Test Loss: {loss}, Test Accuracy: {accuracy * 100:.2f}%")

    # Test OCR
    logger.info("Testing OCR on test dataset")
    ocr = MedicalReportOCR()
    output_dir = "data/processed/OCR_Results"
    csv_path = ocr.test_model_on_dataset(f"{data_config['output_dir']}/test", output_dir)
    logger.info(f"OCR results saved to {csv_path}")

if __name__ == "__main__":
    main()