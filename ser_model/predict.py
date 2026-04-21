"""
Prediction Module
Makes predictions on single audio files
"""

import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, List
import librosa

from config import (
    BEST_MODEL_PATH, EMOTIONS, IDX_TO_EMOTION, SAMPLE_RATE,
    N_MELS, N_FFT, HOP_LENGTH, TARGET_DURATION, LOG_LEVEL,
    INFERENCE_CONFIDENCE_THRESHOLD, RETURN_TOP_K
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class SERPredictor:
    """Handles inference on audio files"""
    
    def __init__(self, model_path: Path = BEST_MODEL_PATH, custom_objects: Dict = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            custom_objects: Custom objects for model loading
        """
        self.model_path = Path(model_path)
        
        # Load custom objects if needed
        if custom_objects is None:
            from model_builder import MultiHeadAttention
            custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects=custom_objects
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.sample_rate = SAMPLE_RATE
        self.n_mels = N_MELS
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.target_length = int(TARGET_DURATION * SAMPLE_RATE // HOP_LENGTH)
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {audio_path} (duration: {len(y)/sr:.2f}s)")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def extract_features(self, y: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features"""
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or trim
        if mel_spec_db.shape[1] >= self.target_length:
            start_idx = (mel_spec_db.shape[1] - self.target_length) // 2
            mel_spec_db = mel_spec_db[:, start_idx:start_idx + self.target_length]
        else:
            pad_total = self.target_length - mel_spec_db.shape[1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (pad_left, pad_right)),
                                mode='constant', constant_values=-80)
        
        # Add channel dimension
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        
        return mel_spec_db
    
    def predict_single(self, audio_path: str) -> Dict:
        """
        Make prediction on single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with predictions and confidence
        """
        logger.info(f"\nProcessing: {audio_path}")
        
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        # Extract features
        mel_spec = self.extract_features(y)
        
        # Add batch dimension
        X = np.expand_dims(mel_spec, axis=0)
        
        # Make prediction
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Get results
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        # Get top K predictions
        top_k_indices = np.argsort(predictions)[-RETURN_TOP_K:][::-1]
        top_k_results = [
            {
                'emotion': EMOTIONS[idx],
                'confidence': float(predictions[idx])
            }
            for idx in top_k_indices
        ]
        
        result = {
            'audio_file': str(audio_path),
            'predicted_emotion': EMOTIONS[predicted_idx],
            'confidence': float(confidence),
            'threshold_met': float(confidence) >= INFERENCE_CONFIDENCE_THRESHOLD,
            'all_predictions': {
                EMOTIONS[i]: float(predictions[i])
                for i in range(len(EMOTIONS))
            },
            'top_k_predictions': top_k_results
        }
        
        return result
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        logger.info(f"Making predictions on {len(audio_paths)} audio files...")
        
        for audio_path in audio_paths:
            try:
                result = self.predict_single(audio_path)
                results.append(result)
                logger.info(f"  ✓ {audio_path}: {result['predicted_emotion']} "
                           f"({result['confidence']:.4f})")
            except Exception as e:
                logger.error(f"  ✗ Error processing {audio_path}: {e}")
                results.append({
                    'audio_file': str(audio_path),
                    'error': str(e)
                })
        
        return results
    
    def print_prediction(self, result: Dict):
        """Pretty print prediction result"""
        print("\n" + "="*70)
        print(f"Audio File: {result['audio_file']}")
        print("="*70)
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Confidence Threshold Met: {result['threshold_met']}")
        
        print("\nAll Predictions:")
        for emotion, confidence in sorted(result['all_predictions'].items(),
                                         key=lambda x: x[1], reverse=True):
            bar_length = int(confidence * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"  {emotion:12s}: {bar} {confidence:.4f}")
        
        print("\nTop K Predictions:")
        for i, pred in enumerate(result['top_k_predictions'], 1):
            print(f"  {i}. {pred['emotion']:12s}: {pred['confidence']:.4f}")
        
        print("="*70 + "\n")


def load_and_predict(audio_path: str, model_path: Path = BEST_MODEL_PATH) -> Dict:
    """
    Convenience function to load model and make prediction
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        
    Returns:
        Prediction result
    """
    predictor = SERPredictor(model_path)
    return predictor.predict_single(audio_path)


if __name__ == '__main__':
    # Example usage
    print("Prediction module loaded successfully!")
    print("Use SERPredictor class to make predictions on audio files.")
