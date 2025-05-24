from fastapi import FastAPI, File, UploadFile
from src.models.densenet import XRayDenseNet
from src.data.preprocessing import preprocess_image
import numpy as np
from PIL import Image
import io
import os
from src.utils.logging import setup_logging
from src.utils.config import load_config

app = FastAPI()
logger = setup_logging()
config = load_config()
model = XRayDenseNet(
    input_shape=tuple(config['model']['input_shape']),
    num_classes=config['model']['num_classes'],
    model_path=os.path.join(config['model']['model_dir'], 'final_densenet_model.keras')
)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return {"error": "Invalid file format. Only PNG, JPG, JPEG allowed."}
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = np.array(image)
        image = preprocess_image(image, target_size=config['data']['image_size'])
        prediction = model.model.predict(np.expand_dims(image, axis=0))
        classes = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        result = {cls: float(prob) for cls, prob in zip(classes, prediction[0])}
        logger.info(f"Classified {file.filename}: {result}")
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Error classifying {file.filename}: {str(e)}")
        return {"error": str(e)}