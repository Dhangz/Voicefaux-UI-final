from django.urls import path
from .views import classify_audio, classify_batch_audio

urlpatterns = [
    path('classify/', classify_audio, name='classify_audio'),
    path('classify-batch/', classify_batch_audio, name='classify_batch_audio'),
]
