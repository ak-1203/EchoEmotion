"""
Speech Emotion Recognition (SER) - Production-Ready Implementation
Complete modular system for emotion recognition from audio data
"""

__version__ = "1.0.0"
__author__ = "EchoEmotion"
__description__ = "Speech Emotion Recognition using CNN+BiLSTM+Attention (PyTorch)"

from .config import (
    EMOTIONS,
    NUM_CLASSES,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    DATA_DIR,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
)
from .data_processor import DataLoader, AudioProcessor, DataAugmentor
from .model_builder import SERModel, build_ser_model, build_training_components
from .train_model import train_model, evaluate_training
from .evaluate_model import ModelEvaluator, generate_evaluation_report
from .predict import SERPredictor, load_and_predict

__all__ = [
    'EMOTIONS',
    'NUM_CLASSES',
    'EMOTION_TO_IDX',
    'IDX_TO_EMOTION',
    'BEST_MODEL_PATH',
    'FINAL_MODEL_PATH',
    'DATA_DIR',
    'SAMPLE_RATE',
    'N_MELS',
    'N_FFT',
    'HOP_LENGTH',
    'DataLoader',
    'AudioProcessor',
    'DataAugmentor',
    'SERModel',
    'build_ser_model',
    'build_training_components',
    'train_model',
    'evaluate_training',
    'ModelEvaluator',
    'generate_evaluation_report',
    'SERPredictor',
    'load_and_predict',
]
