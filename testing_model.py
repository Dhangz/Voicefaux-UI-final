import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.display
import noisereduce as nr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms

SR = 16000
DURATION = 5
SAMPLES = SR * DURATION
IMG_SIZE = 224

CLASS_NAMES = ["modified", "unmodified", "synthetic", "spliced"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.model = tv_models.resnet50(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


def preprocess_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y = librosa.util.fix_length(y, size=SAMPLES)
    y = nr.reduce_noise(y=y, sr=SR)
    return y


def create_melspectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=216, axis=1)

    # Convert to 3-channel (RGB)
    mel_rgb = np.stack([mel_db] * 3, axis=-1)
    return mel_rgb


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class AudioPredictor:
    def __init__(self, model_path="model/resnet50.pth"):
        self.model = AudioClassifier().to(device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.eval()

    def predict(self, audio_path):
        y = preprocess_audio(audio_path)
        mel = create_melspectrogram(y)

        x = transform(mel).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))

        return {
            "predicted_class": CLASS_NAMES[pred],
            "confidence": float(probs[pred]),
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }


def visualize(audio_path, result):
    y = preprocess_audio(audio_path)
    mel = librosa.feature.melspectrogram(y=y, sr=SR)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db, sr=SR, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")

    plt.title(
        f"Prediction: {result['predicted_class']} "
        f"({result['confidence']*100:.2f}%)"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predictor = AudioPredictor("model/resnet50.pth")


    predictor.predict

    
    audio_file = r"C:\Users\USER\Desktop\thesis 2 system\Thesis System Final 2\TEST_MODEL\05-27-2025 05.59 pm.wav"

    if os.path.exists(audio_file):
        result = predictor.predict(audio_file)

        print("\nPrediction Result:")
        for k, v in result.items():
            print(f"{k}: {v}")

        visualize(audio_file, result)
    else:
        print("Audio file not found.")
