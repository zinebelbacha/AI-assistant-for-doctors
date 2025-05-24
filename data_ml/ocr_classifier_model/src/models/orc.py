from PIL import Image
import pytesseract
import pandas as pd
import os
from src.data.preprocessing import preprocess_image, enhance_image

class MedicalReportOCR:
    def __init__(self, tesseract_config="--oem 3 --psm 6"):
        self.tesseract_config = tesseract_config

    def extract_text(self, image_path):
        processed_image = preprocess_image(image_path)
        enhanced_image = enhance_image(processed_image)
        pil_image = Image.fromarray(enhanced_image)
        text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
        extracted_data = {
            'full_text': text,
            'measurements': [],
            'values': [],
            'units': []
        }
        lines = text.splitlines()
        for line in lines:
            if any(char.isdigit() for char in line):
                parts = line.split()
                for part in parts:
                    if any(char.isdigit() for char in part):
                        extracted_data['values'].append(part)
                    elif any(unit in part.lower() for unit in ['mg', 'ml', 'g', 'l']):
                        extracted_data['units'].append(part)
        return extracted_data

    def test_model_on_dataset(self, test_dir, output_dir):
        results = []
        os.makedirs(output_dir, exist_ok=True)

        for image_file in os.listdir(test_dir):
            image_path = os.path.join(test_dir, image_file)
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            extracted_data = self.extract_text(image_path)
            results.append({
                'filename': image_file,
                'full_text': extracted_data['full_text'],
                'values': ' | '.join(extracted_data['values']),
                'units': ' | '.join(extracted_data['units'])
            })

            text_file_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_full.txt")
            with open(text_file_path, 'w') as f:
                f.write(extracted_data['full_text'])

        csv_output_path = os.path.join(output_dir, 'extracted_data.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_output_path, index=False)
        return csv_output_path