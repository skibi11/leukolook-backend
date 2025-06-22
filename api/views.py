# api/views.py (New, Rewritten Version)

import logging
import cv2
import numpy as np
import base64
import io

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from inference_sdk import InferenceHTTPClient

logger = logging.getLogger(__name__)

# --- Roboflow and Model Configuration ---
CLIENT_EYES = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=settings.ROBOFLOW_API_KEY)
CLIENT_FACE = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=settings.ROBOFLOW_API_KEY) # Corrected URL from your original code
CONF_THRESHOLD = 0.75
ENHANCED_SIZE = (224, 224)


# --- Helper Functions ---

def to_base64(image_array):
    """Converts a cv2 image (numpy array) to a base64 string."""
    _, buffer = cv2.imencode(".jpg", image_array)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

def enhance_image_unsharp_mask(image, strength=0.5, radius=5):
    """Sharpens the image."""
    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    return cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)


# --- Main API View ---

class EyeDetectionView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        img_file = request.FILES.get("image")
        if not img_file:
            return Response({"warnings": ["No image provided."]}, status=400)

        try:
            # --- Step 1: Read image into memory ---
            # Read the uploaded image file directly into a numpy array for OpenCV
            in_memory_file = io.BytesIO(img_file.read())
            file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
            original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if original_image is None:
                return Response({"warnings": ["Could not decode image."]}, status=400)

            # --- Step 2: Face Detection (using in-memory image) ---
            face_resp = CLIENT_FACE.infer(original_image, model_id="face-detector-v4liw/2")
            if not face_resp.get("predictions"):
                return Response({"warnings": ["No face detected."]}, status=200)

            # --- Step 3: Eye Detection (using in-memory image) ---
            eye_resp = CLIENT_EYES.infer(original_image, model_id="eye-detection-kso3d/3")
            eye_predictions = [p for p in eye_resp.get("predictions", []) if p.get("confidence", 0) > CONF_THRESHOLD]

            if len(eye_predictions) != 2:
                return Response({
                    "original": to_base64(original_image),
                    "two_eyes": None,
                    "leukocoria": None,
                    "warnings": ["Exactly two eyes not detected."]
                }, status=200)

            # --- Step 4: Process Eyes (all in memory) ---
            sorted_eyes = sorted(eye_predictions, key=lambda p: p["x"])
            base64_eyes, flags = {}, {}

            # This import should be inside the function to avoid loading the model globally
            from .model_loader import predict_leukocoria_with_mobilenet

            for side, pred in zip(("left", "right"), sorted_eyes):
                # Crop the eye from the original image
                x1 = int(pred['x'] - pred['width'] / 2)
                y1 = int(pred['y'] - pred['height'] / 2)
                x2 = int(pred['x'] + pred['width'] / 2)
                y2 = int(pred['y'] + pred['height'] / 2)
                eye_crop = original_image[y1:y2, x1:x2]

                if eye_crop.size > 0:
                    # Enhance and resize for the leukocoria model
                    enh = enhance_image_unsharp_mask(eye_crop)
                    enh_rs = cv2.resize(enh, ENHANCED_SIZE)

                    # Get prediction from the local model
                    has_leuko = predict_leukocoria_with_mobilenet(enh_rs)
                    flags[side] = has_leuko

                    # Encode the processed eye crop to base64 for display
                    base64_eyes[side] = to_base64(eye_crop)
                else:
                    flags[side] = None
                    base64_eyes[side] = None
            
            # --- Step 5: Return all results in the API response ---
            # No data is saved to the database or disk.
            return Response({
                "id": None, # No database ID
                "original": to_base64(original_image),
                "two_eyes": base64_eyes,
                "leukocoria": flags,
                "warnings": []
            }, status=200)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return Response({"warnings": [f"An unexpected error occurred: {e}"]}, status=500)