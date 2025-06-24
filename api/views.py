# api/views.py (Final Corrected Version)

import requests
import os
import base64 # <--- IMPORT THIS
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

class EyeDetectionView(APIView):
    permission_classes = [AllowAny] 

    def post(self, request, *args, **kwargs):
        # 1. Get the image from the frontend's request
        image_file = request.data.get('image')
        if not image_file:
            return Response({'error': 'No image provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # 2. Prepare the request for your Gradio API on Hugging Face
        api_url = "https://skibi11-leukolook-api.hf.space/detect/"
        
        # --- START OF CHANGES ---

        # Public Gradio APIs don't need an Authorization header
        headers = {} 

        # Read the image data and encode it in Base64
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Create the JSON payload in the format Gradio expects
        payload = {
            "data": [
                f"data:image/jpeg;base64,{image_base64}"
            ]
        }
        
        # --- END OF CHANGES ---

        # 3. Call the Hugging Face API and handle the response
        try:
            print("Sending request to Hugging Face Gradio API...")
            
            # Send the request using the 'json' parameter instead of 'data'
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            print("Received successful response from Hugging Face.")
            # 4. Send the successful prediction back to your frontend
            return Response(response.json(), status=response.status_code)

        except requests.exceptions.RequestException as e:
            print(f"Error calling Hugging Face API: {e}")
            error_details = f"Service temporarily unavailable: {e}"
            try:
                error_details = e.response.json().get('error', str(e))
            except:
                pass
            
            return Response({'error': error_details}, status=status.HTTP_503_SERVICE_UNAVAILABLE)