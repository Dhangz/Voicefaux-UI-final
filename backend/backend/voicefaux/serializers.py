from rest_framework import serializers

class AudioUploadSerializer(serializers.Serializer):
    audio_file = serializers.FileField()
    
    def validate_audio_file(self, value):
        """Validate audio file format - only .wav files supported"""
        file_name = value.name.lower()
        
        if not file_name.endswith('.wav'):
            raise serializers.ValidationError(
                "Unsupported file format. Only .wav files are supported."
            )
        
        # Check file size (max 50MB for audio files)
        max_size = 50 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File size exceeds maximum limit of {max_size / (1024*1024)}MB"
            )
        
        return value
