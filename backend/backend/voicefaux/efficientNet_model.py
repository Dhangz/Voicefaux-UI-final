import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import torch
import torchvision.models as tv_models
from torchvision import transforms
from torchvision.transforms import ToPILImage
from django.conf import settings
import tempfile
import timm
import torch.nn as nn
# =========================
# CONFIGURATION
# =========================
SR = 16000
DURATION = 5
SAMPLES = SR * DURATION
IMG_SIZE = 224
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

CLASS_NAMES = ["modified", "unmodified", "synthetic", "spliced"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"VoiceFaux ML Model using device: {device}")

# =========================
# MODEL DEFINITION
# =========================
class AudioClassifier(nn.Module):
    """TIMM EfficientNet-B0 audio classifier"""
    def __init__(self, num_classes=4):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# AUDIO PREPROCESSING
# =========================
def preprocess_audio(audio_path):
    """
    Load audio, resample, mono, and fix length
    """
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = librosa.util.fix_length(y, size=SAMPLES)
        return y
    except Exception as e:
        raise ValueError(f"Audio preprocessing error: {e}")

# =========================
# MEL SPECTROGRAM
# =========================
def create_melspectrogram(y):
    """
    Convert waveform to normalized 3-channel mel spectrogram
    """
    try:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # Fix time dimension dynamically
        target_frames = int(np.ceil(SAMPLES / HOP_LENGTH))
        mel_db = librosa.util.fix_length(mel_db, size=target_frames, axis=1)

        # Convert to 3-channel image
        mel_rgb = np.stack([mel_db] * 3, axis=-1).astype(np.float32)

        return mel_rgb
    except Exception as e:
        raise ValueError(f"Mel spectrogram error: {e}")

# =========================
# PREDICTOR CLASS
# =========================
class AudioPredictor:
    """EfficientNet-B0 inference handler"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                settings.BASE_DIR,
                "voicefaux",
                "model",
                "efficientnet_b0.pth"
            )

        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = AudioClassifier(num_classes=len(CLASS_NAMES)).to(device)

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            print(f"✓ EfficientNet-B0 loaded from {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, audio_file):
        """
        Run inference on Django UploadedFile
        """
        temp_path = None
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                for chunk in audio_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name

            # Audio → Spectrogram
            y = preprocess_audio(temp_path)
            mel = create_melspectrogram(y)

            # Tensor
            x = transform(mel).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))

            return {
                "predicted_class": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "all_probabilities": {
                    CLASS_NAMES[i]: round(float(probs[i]), 4)
                    for i in range(len(CLASS_NAMES))
                }
            }

        except Exception as e:
            raise RuntimeError(f"Inference error: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

# =========================
# SINGLETON ACCESS
# =========================
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = AudioPredictor()
    return _predictor

def predict_audio(audio_file):
    """
    Convenience function for views.py
    """
    predictor = get_predictor()
    return predictor.predict(audio_file)
