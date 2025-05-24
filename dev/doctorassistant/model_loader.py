import os
import tensorflow as tf
from django.conf import settings

# les chemins des modèles dans le répertoire mlmodels
OCR_MODEL_PATH = os.path.join(settings.BASE_DIR, "mlmodels", "ocr_model.h5")
PNORTUB_MODEL_PATH = os.path.join(settings.BASE_DIR, "mlmodels", "final_densenet_model.h5")

# Chargement les modèles
ocr_model = tf.keras.models.load_model(OCR_MODEL_PATH)
pnortub_model = tf.keras.models.load_model(PNORTUB_MODEL_PATH)

def get_ocr_model():
    return ocr_model

def get_pnortub_model():
    return pnortub_model
