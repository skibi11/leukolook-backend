import os
import tensorflow as tf
import numpy as np

# Define absolute model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "MobileNetV1_best.keras")

# Load the model once at module level
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

def predict_leukocoria_with_mobilenet(image: np.ndarray) -> bool:
    """
    Predict leukocoria using MobileNetV1_best.keras.
    Expects a 224x224 RGB image (uint8 or float32).
    Returns True if leukocoria is detected, False otherwise.
    """
    if image.shape != (224, 224, 3):
        raise ValueError(f"Expected image of shape (224, 224, 3), got {image.shape}")

    img = image.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

    try:
        prediction = model.predict(img, verbose=0)[0][0]
        return prediction > 0.5
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
