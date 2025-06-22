import requests
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny # Import AllowAny

class EyeDetectionView(APIView):
    # Allow any request to come through, since we are not handling users yet.
    permission_classes = [AllowAny] 

    def post(self, request, *args, **kwargs):
        # 1. Get the image from the frontend's request
        image_file = request.data.get('image')
        if not image_file:
            return Response({'error': 'No image provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # 2. Prepare the request for Hugging Face
        # Get the secret token from environment variables
        api_key = os.environ.get('HUGGING_FACE_TOKEN')
        # The URL for your specific model's Inference API
        api_url = "https://api-inference.huggingface.co/models/skibi11/leukolook-eye-detector"
        headers = {"Authorization": f"Bearer {api_key}"}

        # Read the binary data from the uploaded file
        image_data = image_file.read()

        # 3. Call the Hugging Face API and handle the response
        try:
            print("Sending request to Hugging Face API...")
            response = requests.post(api_url, headers=headers, data=image_data)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            print("Received successful response from Hugging Face.")
            # 4. Send the successful prediction from Hugging Face back to your frontend
            return Response(response.json(), status=response.status_code)

        except requests.exceptions.RequestException as e:
            # This block runs if the request fails
            print(f"Error calling Hugging Face API: {e}")
            error_details = f"Service temporarily unavailable: {e}"
            try:
                # Try to get a more specific error message from the API's response
                error_details = e.response.json().get('error', str(e))
            except:
                pass

            return Response({'error': error_details}, status=status.HTTP_503_SERVICE_UNAVAILABLE)