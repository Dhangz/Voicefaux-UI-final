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

class BatchAudioUploadSerializer(serializers.Serializer):
    audio_files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )
    
    def validate_audio_files(self, value):
        # Validate each file
        for audio_file in value:
            if not audio_file.name.endswith('.wav'):
                raise serializers.ValidationError(
                    f"File {audio_file.name} is not a .wav file. Only .wav files are supported"
                )
        
        # Optional: Add a limit to prevent overwhelming the server
        max_files = 20
        if len(value) > max_files:
            raise serializers.ValidationError(
                f"Maximum {max_files} files allowed per batch"
            )
        
        return value