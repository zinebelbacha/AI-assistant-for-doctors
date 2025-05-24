from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import os
import random

class SputumReportGenerator:
    def __init__(self, output_dir, image_size=(800, 600), font_path=None):
        self.output_dir = output_dir
        self.image_size = image_size
        self.fake = Faker('fr_FR')
        self.font = self._load_font(font_path)
        os.makedirs(output_dir, exist_ok=True)
        self._create_directories()

    def _load_font(self, font_path):
        try:
            return ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            print("Using default font.")
            return ImageFont.load_default()

    def _create_directories(self):
        splits = ['train', 'test', 'val']
        categories = ['normal', 'pneumonie', 'tuberculose']
        for split in splits:
            for category in categories:
                os.makedirs(os.path.join(self.output_dir, split, category), exist_ok=True)

    def generate_cytology_data(self, category):
        if category == 'normal':
            return {
                "Cellules squameuses": f"{random.randint(0, 10)} / HPF",
                "Cellules inflammatoires": f"{random.randint(10, 30)} / HPF",
                "Cellules atypiques": "Absentes",
                "Bactéries": "Absentes",
                "Champignons": "Absents",
                "Conclusion": "Aucun signe de malignité détecté."
            }
        elif category == 'pneumonie':
            return {
                "Cellules squameuses": f"{random.randint(5, 15)} / HPF",
                "Cellules inflammatoires": f"{random.randint(50, 100)} / HPF",
                "Cellules atypiques": random.choice(["Présentes", "Absentes"]),
                "Bactéries": "Présentes",
                "Champignons": random.choice(["Absents", "Présents"]),
                "Conclusion": "Présence d'une inflammation sévère et de bactéries; suspicion de pneumonie."
            }
        elif category == 'tuberculose':
            return {
                "Cellules squameuses": f"{random.randint(10, 20)} / HPF",
                "Cellules inflammatoires": f"{random.randint(70, 120)} / HPF",
                "Cellules atypiques": "Présentes",
                "Bactéries": "Absentes",
                "Champignons": "Absents",
                "Conclusion": "Présence de cellules atypiques; nécessitant un suivi pour tuberculose."
            }

    def generate_report(self, index, split, category):
        image = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(image)
        patient_name = self.fake.name()
        patient_id = self.fake.uuid4()[:8]
        date_of_birth = self.fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%d/%m/%Y")
        report_date = self.fake.date_this_year().strftime("%d/%m/%Y")
        cytology_results = self.generate_cytology_data(category)

        x, y = 50, 50
        draw.text((x, y), f"Rapport de Cytologie - ID: {patient_id}", fill="black", font=self.font)
        y += 40
        draw.text((x, y), f"Nom du patient : {patient_name}", fill="black", font=self.font)
        y += 30
        draw.text((x, y), f"Date de naissance : {date_of_birth}", fill="black", font=self.font)
        y += 30
        draw.text((x, y), f"Date du rapport : {report_date}", fill="black", font=self.font)
        y += 40
        draw.text((x, y), "Résultats de cytologie de l'expectoration :", fill="black", font=self.font)
        y += 30
        for test, result in cytology_results.items():
            draw.text((x, y), f"{test} : {result}", fill="black", font=self.font)
            y += 30

        image_path = os.path.join(self.output_dir, split, category, f"sputum_cytology_report_{index}.png")
        image.save(image_path)
        return image_path

    def generate_dataset(self, num_reports, train_ratio=0.6, test_ratio=0.2):
        for i in range(num_reports):
            split = 'train' if i < train_ratio * num_reports else 'test' if i < (train_ratio + test_ratio) * num_reports else 'val'
            category = random.choice(['normal', 'pneumonie', 'tuberculose'])
            self.generate_report(i, split, category)