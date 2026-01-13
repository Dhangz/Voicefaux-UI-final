from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import AudioUploadSerializer, BatchAudioUploadSerializer
from .efficientNet_model import predict_audio

import torch
import traceback
from .efficientNet_model import device, CLASS_NAMES, SR, DURATION

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


@api_view(['POST'])
def classify_batch_audio(request):
    """
    Batch audio classification endpoint
    
    POST /api/classify-batch/
    Body: multipart/form-data with 'audio_files' field (multiple files)
    
    Returns:
        {
            "success": true,
            "data": {
                "total_files": 3,
                "successful": 3,
                "failed": 0,
                "results": [
                    {
                        "filename": "audio1.wav",
                        "predicted_class": "synthetic",
                        "confidence": 0.95,
                        "all_probabilities": {
                            "modified": 2.15,
                            "unmodified": 1.23,
                            "synthetic": 95.42,
                            "spliced": 1.20
                        }
                    },
                    ...
                ],
                "errors": []
            }
        }
    """
    serializer = BatchAudioUploadSerializer(data=request.data)
    
    if serializer.is_valid():
        audio_files = serializer.validated_data['audio_files']
        
        results = []
        errors = []
        successful = 0
        failed = 0
        
        for audio_file in audio_files:
            try:
                # Perform classification for each file
                result = predict_audio(audio_file)
                
                # Add filename to result
                result['filename'] = audio_file.name
                results.append(result)
                successful += 1
                
            except Exception as e:
                # Log the error but continue processing other files
                error_msg = f"Failed to process {audio_file.name}: {str(e)}"
                print(f"❌ {error_msg}")
                print(traceback.format_exc())
                
                errors.append({
                    "filename": audio_file.name,
                    "error": str(e)
                })
                failed += 1
        
        return Response(
            {
                "success": True,
                "data": {
                    "total_files": len(audio_files),
                    "successful": successful,
                    "failed": failed,
                    "results": results,
                    "errors": errors
                }
            }, 
            status=status.HTTP_200_OK
        )
    
    return Response(
        {
            "success": False,
            "errors": serializer.errors
        }, 
        status=status.HTTP_400_BAD_REQUEST
    )

