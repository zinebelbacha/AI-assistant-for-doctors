from src.data.dataset import XRayDataset
from src.utils.config import load_config
from src.utils.logging import setup_logging

def main():
    logger = setup_logging()
    config = load_config()
    data_config = config['data']

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
    logger.info("Data extraction and organization completed")

if __name__ == "__main__":
    main()