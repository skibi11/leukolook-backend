# api/views.py

import os
import time
import requests
import tempfile
import logging
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import viewsets, generics, status, permissions
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.decorators import api_view, permission_classes

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import base64

from .serializers import (
    UserSerializer,
    EmailTokenObtainPairSerializer,
    RegistrationSerializer,
    EyeTestResultSerializer,
)
from .models import EyeTestResult

logger = logging.getLogger(__name__)
User = get_user_model()

CLIENT_EYES = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="TLZLPiIvbxIDm8zecEE7")
CLIENT_FACE = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key="TLZLPiIvbxIDm8zecEE7")

MAX_INFER_DIM = 1024
CONF_THRESHOLD = 0.75
ENHANCED_SIZE = (224, 224)

# --------------------------- Helpers ---------------------------

def enhance_image_unsharp_mask(image, strength=0.5, radius=5):
    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    return cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)

def detect_faces_roboflow(image_path):
    resp = CLIENT_FACE.infer(image_path, model_id="face-detector-v4liw/2")
    return resp.get("predictions", [])

def detect_eyes_roboflow(image_path, scale_w=2, scale_h=5):
    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"Cannot read {image_path!r}")

    h, w = raw.shape[:2]
    scale = min(1.0, MAX_INFER_DIM / max(h, w))
    if scale < 1.0:
        small = cv2.resize(raw, (int(w*scale), int(h*scale)))
        cv2.imwrite("temp_infer.jpg", small)
        resp = CLIENT_EYES.infer("temp_infer.jpg", model_id="eye-detection-kso3d/3")
    else:
        resp = CLIENT_EYES.infer(image_path, model_id="eye-detection-kso3d/3")

    crops = []
    for p in resp.get("predictions", []):
        if p.get("confidence", 0) < CONF_THRESHOLD:
            continue
        cx, cy = p["x"] / scale, p["y"] / scale
        bw, bh = p["width"] / scale, p["height"] / scale
        nw, nh = bw * scale_w, bh * scale_h
        ex1 = max(0, int(cx - nw/2))
        ey1 = max(0, int(cy - nh/2))
        ex2 = min(w, int(cx + nw/2))
        ey2 = min(h, int(cy + nh/2))
        crop = raw[ey1:ey2, ex1:ex2]
        if crop.size:
            crops.append({"coords": (ex1, ey1, ex2, ey2), "image": crop})
    return raw, raw.copy(), crops

def get_largest_iris_prediction(eye_crop):
    cv2.imwrite("temp_crop.jpg", eye_crop)
    resp = CLIENT_EYES.infer("temp_crop.jpg", model_id="iris_120_set/7")
    preds = resp.get("predictions", [])
    return max(preds, key=lambda p: p["width"] * p["height"]) if preds else None

def detect_leukocoria(iris_crop):
    """
    Sends the cropped iris to the FastAPI model server for leukocoria prediction.
    Returns (bool: has_leukocoria, float: confidence)
    """
    if iris_crop is None or iris_crop.size == 0:
        return False, 0.0

    try:
        # Save iris crop temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, iris_crop)

        # Prepare multipart form data
        with open(temp_path, "rb") as f:
            files = {'image': ('iris.jpg', f, 'image/jpeg')}
            response = requests.post("http://127.0.0.1:8080/predict/", files=files)

        if response.status_code == 200:
            data = response.json()
            return data.get("has_leukocoria", False), data.get("prediction", 0.0)
        else:
            logging.error(f"Inference API error: {response.status_code} - {response.text}")
            return False, 0.0

    except Exception as e:
        logging.exception("Failed to call leukocoria inference API")
        return False, 0.0


def to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# -------------------------------------------------------------------
# ViewSets & Simple Endpoints
# -------------------------------------------------------------------


@api_view(['GET', 'PATCH'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    print("Authenticated as:", request.user)
    user = request.user

    if request.method == 'GET':
        serializer = UserSerializer(user)
        return Response(serializer.data)

    if request.method == 'PATCH':
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [permissions.IsAdminUser]
    permission_classes = [AllowAny]
    

class HelloView(APIView):
    permission_classes = [AllowAny]
    def get(self, request):
        return Response({"message": "Hello, Django!"})

class EmailTokenObtainPairView(TokenObtainPairView):
    serializer_class = EmailTokenObtainPairSerializer

class RegisterView(generics.CreateAPIView):
    serializer_class = RegistrationSerializer
    permission_classes = [AllowAny]


class EyeDetectionView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        img_file = request.FILES.get("image")
        if not img_file:
            return Response({"warnings": ["No image provided."]}, status=400)

        # Directory setup
        save_dir = os.path.join(settings.MEDIA_ROOT, "eye_tests")
        iris_dir = os.path.join(save_dir, "iris_crops")
        eye_dir  = os.path.join(save_dir, "two_eyes")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(iris_dir, exist_ok=True)
        os.makedirs(eye_dir, exist_ok=True)

        # Save original file
        ts = int(time.time() * 1000)
        name = f"{ts}_{img_file.name}"
        path = os.path.join(save_dir, name)
        with open(path, "wb") as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        original_url = request.build_absolute_uri(
            os.path.join(settings.MEDIA_URL, "eye_tests", name)
        )

        # Face check
        if not detect_faces_roboflow(path):
            return Response({"warnings": ["No face detected."]}, status=200)

        # Eye detection
        _, _, eye_crops = detect_eyes_roboflow(path)
        if len(eye_crops) != 2:
            return Response({
                "original": original_url,
                "two_eyes": None,
                "leukocoria": None,
                "warnings": ["Exactly two eyes not detected."]
            }, status=200)

        # Process eyes
        sorted_eyes = sorted(eye_crops, key=lambda e: e["coords"][0])
        urls_eyes, flags = {}, {}

        for side, info in zip(("left", "right"), sorted_eyes):
            ex1, ey1, ex2, ey2 = info["coords"]
            eye_img = info["image"]

            # Iris detection
            pred = get_largest_iris_prediction(eye_img)
            if pred:
                cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)

                # ⚠️ draw on a separate copy just for saving/display
                display_eye_img = eye_img.copy()
                cv2.rectangle(display_eye_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ✅ do not draw on eye_img used for iris_crop
                iris_crop = eye_img[y1:y2, x1:x2]
                enh = enhance_image_unsharp_mask(iris_crop)
                enh_rs = cv2.resize(enh, ENHANCED_SIZE)

                iris_filename = f"{ts}_iris_{side}_{img_file.name}"
                iris_path = os.path.join(iris_dir, iris_filename)
                cv2.imwrite(iris_path, enh_rs)

                has_leuko, confidence = detect_leukocoria(enh_rs)
                print(f"[Leukocoria] Side: {side}, Has: {has_leuko}, Confidence: {confidence:.4f}")
                flags[side] = has_leuko
            else:
                flags[side] = None

            # Save annotated eye image
            eye_filename = f"{ts}_{side}_{img_file.name}"
            eye_path = os.path.join(eye_dir, eye_filename)
            cv2.imwrite(eye_path, display_eye_img if pred else eye_img)
            urls_eyes[side] = request.build_absolute_uri(
                os.path.join(settings.MEDIA_URL, "eye_tests/two_eyes", eye_filename)
            )
            
            # Assign valid user
            if request.user.is_authenticated:
                assigned_user = request.user
            else:
                try:
                    assigned_user = User.objects.get(username="guest_user")
                except User.DoesNotExist:
                    assigned_user = User.objects.create_user(
                        username="guest_user",
                        email="guest@example.com",
                        password="guestpass123"
                    )

            # ✅ Use assigned_user here instead of request.user
            result = EyeTestResult.objects.create(
                user=assigned_user,
                original=path,
                left_eye=urls_eyes.get("left"),
                right_eye=urls_eyes.get("right"),
                has_leukocoria_left=flags.get("left"),
                has_leukocoria_right=flags.get("right"),
            )

        return Response({
            "id": result.id,
            "original": original_url,
            "two_eyes": urls_eyes,
            "leukocoria": flags,
            "warnings": []
        }, status=200)
        
     
class EyeTestResultViewSet(viewsets.ModelViewSet):
    queryset = EyeTestResult.objects.all()
    serializer_class = EyeTestResultSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(user=self.request.user)
