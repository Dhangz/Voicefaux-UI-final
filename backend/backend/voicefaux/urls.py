from django.urls import path
from .views import classify_audio, model_info

urlpatterns = [
    path('classify/', classify_audio, name='classify_audio'),
    path('model-info/', model_info, name='model_info'),
]
