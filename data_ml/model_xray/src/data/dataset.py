from zipfile import ZipFile, BadZipFile
import os
import shutil
import random
from src.utils.logging import setup_logging

class XRayDataset:
    def __init__(self, zip_file_path, tb_zip_file_path, extraction_dir, normal_dir):
        self.zip_file_path = zip_file_path
        self.tb_zip_file_path = tb_zip_file_path
        self.extraction_dir = extraction_dir
        self.normal_dir = normal_dir
        self.logger = setup_logging()
        self.classes = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

    def extract_zip(self):
        os.makedirs(self.extraction_dir, exist_ok=True)
        try:
            with ZipFile(self.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extraction_dir)
            self.logger.info(f"Extracted {self.zip_file_path} to {self.extraction_dir}")
            with ZipFile(self.tb_zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extraction_dir)
            self.logger.info(f"Extracted {self.tb_zip_file_path} to {self.extraction_dir}")
        except BadZipFile as e:
            self.logger.error(f"Invalid zip file: {e}")
            raise

    def move_normal_images(self, source_dir, num_images=3372):
        os.makedirs(self.normal_dir, exist_ok=True)
        extensions = {'.jpg', '.png', '.jpeg'}
        moved_count = 0
        for filename in os.listdir(source_dir):
            if moved_count >= num_images:
                break
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(self.normal_dir, filename)
            if os.path.isfile(source_path) and any(filename.lower().endswith(ext) for ext in extensions):
                shutil.move(source_path, destination_path)
                moved_count += 1
                self.logger.info(f"Moved: {filename}")
        self.logger.info(f"Total normal images moved: {moved_count}")

    def organize_dataset(self, train_ratio=0.6, test_ratio=0.2):
        splits = {
            'train': train_ratio,
            'test': test_ratio,
            'val': 1 - train_ratio - test_ratio
        }
        for split in splits:
            for cls in self.classes:
                os.makedirs(os.path.join(self.extraction_dir, split, cls), exist_ok=True)

        # Organize Normal images
        normal_files = os.listdir(self.normal_dir)
        random.shuffle(normal_files)
        total = len(normal_files)
        train_end = int(total * splits['train'])
        test_end = train_end + int(total * splits['test'])

        for i, filename in enumerate(normal_files):
            split = 'train' if i < train_end else 'test' if i < test_end else 'val'
            src = os.path.join(self.normal_dir, filename)
            dst = os.path.join(self.extraction_dir, split, 'NORMAL', filename)
            shutil.move(src, dst)

        # Organize Pneumonia and Tuberculosis from extracted dataset
        for cls in ['PNEUMONIA', 'TUBERCULOSIS']:
            src_dir = os.path.join(self.extraction_dir, cls.lower())
            if not os.path.exists(src_dir):
                src_dir = os.path.join(self.extraction_dir, 'TB_Chest_Radiography_Database', cls.capitalize())
            files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.shuffle(files)
            total = len(files)
            train_end = int(total * splits['train'])
            test_end = train_end + int(total * splits['test'])

            for i, filename in enumerate(files):
                split = 'train' if i < train_end else 'test' if i < test_end else 'val'
                src = os.path.join(src_dir, filename)
                dst = os.path.join(self.extraction_dir, split, cls, filename)
                shutil.move(src, dst)

        self.logger.info("Dataset organized into train/test/val splits")

    def count_files(self, directory):
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])