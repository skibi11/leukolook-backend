# config/urls.py (New, Simplified Version)

from django.contrib import admin
from django.urls import path
from api.views import EyeDetectionView  # We only need to import this one view

urlpatterns = [
    path('admin/', admin.site.urls),

    # This is the only API endpoint your application needs now
    path('api/eye-detect/', EyeDetectionView.as_view(), name='eye_detect'),
]