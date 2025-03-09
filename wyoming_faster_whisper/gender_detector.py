"""Gender detector."""
import logging
import wave

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

_LOGGER = logging.getLogger(__name__)


class GenderDetector:
    """Gender detection using wav2vec2 model."""

    def __init__(self, device: str = "cuda"):
        """Initialize gender detector."""
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        _LOGGER.info(f"Initializing gender detector using device: {self.device}")

        model_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect(self, audio_path: str) -> str:
        """Detect gender from audio file."""
        try:
            with wave.open(audio_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)

            audio_data = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Check if there is significant sound
            if np.max(np.abs(audio_data)) <= 0.05:
                _LOGGER.warning("No significant sound detected for gender detection")
                return "unknown"

            inputs = self.feature_extractor(
                audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)

                # Map predicted IDs to labels
                predicted_label = self.model.config.id2label[predicted_ids.item()]

            return predicted_label

        except Exception as e:
            _LOGGER.error(f"Error detecting gender: {str(e)}")
            return "unknown"
