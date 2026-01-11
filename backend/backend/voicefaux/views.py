from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import AudioUploadSerializer
from .ml_model import predict_audio

import torch
from .ml_model import device, CLASS_NAMES, SR, DURATION

@api_view(['POST'])
def classify_audio(request):
    """
    Audio classification endpoint
    
    POST /api/classify/
    Body: multipart/form-data with 'audio_file' field
    
    Returns:
        {
            "success": true,
            "data": {
                "predicted_class": "synthetic",
                "confidence": 0.95,
                "all_probabilities": {
                    "modified": 2.15,
                    "unmodified": 1.23,
                    "synthetic": 95.42,
                    "spliced": 1.20
                }
            }
        }
    """
    serializer = AudioUploadSerializer(data=request.data)
    
    if serializer.is_valid():
        audio_file = serializer.validated_data['audio_file']
        
        try:
            # Perform classification
            result = predict_audio(audio_file)
            
            return Response(
                {
                    "success": True,
                    "data": result
                }, 
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            # Log the full traceback for debugging
            print(f"❌ Error during classification: {str(e)}")
            print(traceback.format_exc())
            
            return Response(
                {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to classify audio file"
                }, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    return Response(
        {
            "success": False,
            "errors": serializer.errors
        }, 
        status=status.HTTP_400_BAD_REQUEST
    )

@api_view(['GET'])
def model_info(request):
    """
    Get model information
    
    GET /api/model-info/
    """
    
    return Response(
        {
            "model_architecture": "ResNet50",
            "num_classes": len(CLASS_NAMES),
            "classes": CLASS_NAMES,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "sample_rate": SR,
            "audio_duration": DURATION,
            "input_format": "audio files (.wav only)"
        },
        status=status.HTTP_200_OK
    )