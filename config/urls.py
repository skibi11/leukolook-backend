from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from rest_framework_simplejwt.views import TokenRefreshView
from api.views import (
    UserViewSet,
    HelloView,
    EmailTokenObtainPairView,
    RegisterView,
    EyeDetectionView,
    EyeTestResultViewSet,
)
from django.conf import settings
from django.conf.urls.static import static
from api.views import current_user_view

router = routers.DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'eye-results', EyeTestResultViewSet, basename='eye-results')

urlpatterns = [
    path('admin/', admin.site.urls),

    # Eye detection
    path('api/eye-detect/', EyeDetectionView.as_view(), name='eye_detect'),

    # Registration endpoint
    path('api/register/', RegisterView.as_view(), name='register'),

    # JWT login endpoints (Email and Password)  
    path('api/token/', EmailTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Other API routes
    path('api/hello/', HelloView.as_view(), name='hello'),
    
    path('api/', include(router.urls)),
    
    path('api/users/me/', current_user_view, name='user_me'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
