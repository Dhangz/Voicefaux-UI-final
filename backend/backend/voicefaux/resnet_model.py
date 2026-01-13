import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import noisereduce as nr
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms
from django.conf import settings
import tempfile

# Audio processing constants
SR = 16000
DURATION = 5
SAMPLES = SR * DURATION
IMG_SIZE = 224

# Class names matching your model
CLASS_NAMES = ["modified", "unmodified", "synthetic", "spliced"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"VoiceFaux ML Model using device: {device}")


class AudioClassifier(nn.Module):
    """ResNet50-based audio classifier"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = tv_models.resnet50(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# Image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_audio(audio_path):
    """
    Load and preprocess audio file
    - Loads at 16kHz sample rate
    - Fixes length to 5 seconds
    - Applies noise reduction
    """
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = librosa.util.fix_length(y, size=SAMPLES)
        y = nr.reduce_noise(y=y, sr=SR)
        return y
    except Exception as e:
        raise ValueError(f"Error preprocessing audio: {str(e)}")


def create_melspectrogram(y):
    """
    Create mel spectrogram from audio waveform
    Returns 3-channel RGB image suitable for ResNet
    """
    try:
        # Generate mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=216, axis=1)
        
        # Convert to 3-channel RGB for ResNet compatibility
        mel_rgb = np.stack([mel_db] * 3, axis=-1)
        
        return mel_rgb
    except Exception as e:
        raise ValueError(f"Error creating mel spectrogram: {str(e)}")


class AudioPredictor:
    """Main prediction class"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                settings.BASE_DIR, 
                'voicefaux', 
                'model', 
                'resnet50.pth'
            )
        
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ResNet50 model"""
        try:
            self.model = AudioClassifier(num_classes=4).to(device)
            
            if os.path.exists(self.model_path):
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=device)
                )
                self.model.eval()
                print(f"✓ Model loaded successfully from {self.model_path}")
            else:
                print(f"⚠ Warning: Model file not found at {self.model_path}")
                print("Using untrained model for demonstration")
                self.model.eval()
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict(self, audio_file):
        """
        Predict audio classification
        
        Args:
            audio_file: Django UploadedFile object
            
        Returns:
            dict with prediction results
        """
        temp_path = None
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                for chunk in audio_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name
            
            # Preprocess audio
            y = preprocess_audio(temp_path)
            mel = create_melspectrogram(y)
            
            # Transform to tensor
            x = transform(mel).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
            
            # Prepare result
            result = {
                "predicted_class": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "all_probabilities": {
                    CLASS_NAMES[i]: round(float(probs[i]), 4)
                    for i in range(len(CLASS_NAMES))
                }
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
        
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


# Global predictor instance (loaded once)
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = AudioPredictor()
    return _predictor


def predict_audio(audio_file):
    """
    Convenience function for prediction
    """
    predictor = get_predictor()
    return predictor.predict(audio_file)