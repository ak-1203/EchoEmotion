"""
Prediction Module
Makes predictions on single/batch audio files (PyTorch)
"""

import logging
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch

from config import (
    BEST_MODEL_PATH,
    EMOTIONS,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    TARGET_DURATION,
    LOG_LEVEL,
    INFERENCE_CONFIDENCE_THRESHOLD,
    RETURN_TOP_K,
)
from model_builder import build_ser_model

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class SERPredictor:
    """Handles inference on audio files."""

    def __init__(self, model_path: Path = BEST_MODEL_PATH, device: torch.device = None):
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path} on {self.device}...")
        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        input_shape = checkpoint.get('input_shape')
        if input_shape is None:
            # fallback for backward compatibility
            input_shape = (1, N_MELS, int(TARGET_DURATION * SAMPLE_RATE // HOP_LENGTH))

        self.model = build_ser_model(tuple(input_shape)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.scaler_mean = checkpoint.get('scaler_mean')
        self.scaler_scale = checkpoint.get('scaler_scale')

        self.sample_rate = SAMPLE_RATE
        self.n_mels = N_MELS
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.target_length = int(TARGET_DURATION * SAMPLE_RATE // HOP_LENGTH)

    def load_audio(self, audio_path: str):
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        logger.info(f"Loaded audio: {audio_path} (duration: {len(y) / sr:.2f}s)")
        return y, sr

    def extract_features(self, y: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] >= self.target_length:
            start_idx = (mel_spec_db.shape[1] - self.target_length) // 2
            mel_spec_db = mel_spec_db[:, start_idx:start_idx + self.target_length]
        else:
            pad_total = self.target_length - mel_spec_db.shape[1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-80)

        # Optional global scaler normalization if available
        if self.scaler_mean is not None and self.scaler_scale is not None:
            flat = mel_spec_db.reshape(-1)
            mean = np.array(self.scaler_mean, dtype=np.float32)
            scale = np.array(self.scaler_scale, dtype=np.float32)
            if flat.shape[0] == mean.shape[0]:
                mel_spec_db = ((flat - mean) / np.clip(scale, a_min=1e-8, a_max=None)).reshape(self.n_mels, self.target_length)
            else:
                mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        else:
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return mel_spec_db.astype(np.float32)

    def predict_single(self, audio_path: str) -> Dict:
        logger.info(f"\nProcessing: {audio_path}")

        y, _ = self.load_audio(audio_path)
        mel_spec = self.extract_features(y)

        # Model expects (B, C, n_mels, time)
        X = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(X)
            predictions = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])

        top_k_indices = np.argsort(predictions)[-RETURN_TOP_K:][::-1]
        top_k_results = [{'emotion': EMOTIONS[idx], 'confidence': float(predictions[idx])} for idx in top_k_indices]

        return {
            'audio_file': str(audio_path),
            'predicted_emotion': EMOTIONS[predicted_idx],
            'confidence': confidence,
            'threshold_met': confidence >= INFERENCE_CONFIDENCE_THRESHOLD,
            'all_predictions': {EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))},
            'top_k_predictions': top_k_results,
        }

    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        results = []
        logger.info(f"Making predictions on {len(audio_paths)} audio files...")

        for audio_path in audio_paths:
            try:
                result = self.predict_single(audio_path)
                results.append(result)
                logger.info(f"  OK {audio_path}: {result['predicted_emotion']} ({result['confidence']:.4f})")
            except Exception as exc:
                logger.error(f"  Error processing {audio_path}: {exc}")
                results.append({'audio_file': str(audio_path), 'error': str(exc)})

        return results

    def print_prediction(self, result: Dict):
        print('\n' + '=' * 70)
        print(f"Audio File: {result['audio_file']}")
        print('=' * 70)
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
        print(f"Confidence Threshold Met: {result['threshold_met']}")

        print('\nAll Predictions:')
        for emotion, confidence in sorted(result['all_predictions'].items(), key=lambda x: x[1], reverse=True):
            bar_length = int(confidence * 30)
            bar = '#' * bar_length + '.' * (30 - bar_length)
            print(f"  {emotion:12s}: {bar} {confidence:.4f}")

        print('\nTop K Predictions:')
        for i, pred in enumerate(result['top_k_predictions'], 1):
            print(f"  {i}. {pred['emotion']:12s}: {pred['confidence']:.4f}")

        print('=' * 70 + '\n')


def load_and_predict(audio_path: str, model_path: Path = BEST_MODEL_PATH) -> Dict:
    predictor = SERPredictor(model_path)
    return predictor.predict_single(audio_path)


if __name__ == '__main__':
    print('Prediction module loaded successfully!')
    print('Use SERPredictor class to make predictions on audio files.')
