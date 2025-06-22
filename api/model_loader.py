# api/model_loader.py (New Lazy-Loading Version)

import os
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Model Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "MobileNetV1_best.keras")

# --- Global Model Variable ---
# We initialize the model as None. It will be loaded only when needed.
model = None

def predict_leukocoria_with_mobilenet(image: np.ndarray) -> bool:
    """
    Predicts leukocoria, loading the MobileNet model into memory on the first run.
    Expects a 224x224 RGB image.
    """
    global model

    # --- LAZY LOADING BLOCK ---
    # If the model hasn't been loaded yet, load it now.
    if model is None:
        logger.info(f"Model not loaded. Loading model from: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            # If loading fails, raise a critical error.
            logger.error(f"CRITICAL: Failed to load model at {MODEL_PATH}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    # --- END LAZY LOADING BLOCK ---

    if image.shape != (224, 224, 3):
        raise ValueError(f"Expected image of shape (224, 224, 3), got {image.shape}")

    img = image.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    try:
        prediction = model.predict(img, verbose=0)[0][0]
        return prediction > 0.5
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")