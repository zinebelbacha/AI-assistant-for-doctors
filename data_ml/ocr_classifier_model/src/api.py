from fastapi import FastAPI, File, UploadFile
from src.models.ocr import MedicalReportOCR
from src.models.cnn_model import SputumCNN
import numpy as np
from PIL import Image
import io
import os
from src.utils.logging import setup_logging

app = FastAPI()
logger = setup_logging()
ocr = MedicalReportOCR()
cnn = SputumCNN()

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return {"error": "Invalid file format. Only PNG, JPG, JPEG allowed."}
    try:
        image = Image.open(io.BytesIO(await file.read()))
        temp_path = "temp.png"
        image.save(temp_path)
        result = ocr.extract_text(temp_path)
        os.remove(temp_path)
        logger.info(f"Text extracted from {file.filename}")
        return result
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        return {"error": str(e)}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return {"error": "Invalid file format. Only PNG, JPG, JPEG allowed."}
    try:
        image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
        image_array = np.array(image) / 255.0
        prediction = cnn.model.predict(np.expand_dims(image_array, axis=0))
        logger.info(f"Classification performed on {file.filename}")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Error classifying {file.filename}: {str(e)}")
        return {"error": str(e)}