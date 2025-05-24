from django.urls import path
from doctorassistant.views import PredictView, custom_login_view, register_with_face, index_view
from django.contrib.auth.views import LoginView
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    
    path('admin/', admin.site.urls),
    path('', index_view, name='index'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('login-face/', custom_login_view, name='login_face'),
    path('signup-face/', register_with_face, name='signup_face'),
    
  
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
