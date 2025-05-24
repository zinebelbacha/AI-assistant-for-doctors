from src.data.generator import SputumReportGenerator
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    logger.info("Starting data generation")

    config = load_config()
    data_config = config['data']

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
    logger.info("Data generation completed")

if __name__ == "__main__":
    main()