# in api/views.py

import requests
import os
import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

HF_PIPELINE_URL = "https://skibi11-leukolook-api.hf.space/detect/"

class EyeDetectionView(APIView):
    permission_classes = [AllowAny]
    def post(self, request, *args, **kwargs):
        image_file = request.data.get('image')
        if not image_file:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Read the image data and encode it to Base64
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Create the JSON payload that the HF Space expects
            payload = {"image_base64": image_base64}

            # Send the JSON payload to the Hugging Face Space
            response = requests.post(HF_PIPELINE_URL, json=payload)
            response.raise_for_status()

            # Forward the final JSON result back to the frontend
            return Response(response.json(), status=response.status_code)

        except requests.exceptions.RequestException as e:
            # Handle errors from the Hugging Face service
            try:
              error_content = e.response.json()
            except:
              error_content = {"error": "The analysis service is currently unavailable."}
            return Response(error_content, status=status.HTTP_503_SERVICE_UNAVAILABLE)