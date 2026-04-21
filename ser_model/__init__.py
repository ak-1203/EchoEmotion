"""
Speech Emotion Recognition (SER) - Production-Ready Implementation
Complete modular system for emotion recognition from audio data

Modules:
    - config: Centralized configuration and hyperparameters
    - data_processor: Data loading, preprocessing, augmentation
    - model_builder: CNN+BiLSTM+Attention neural network
    - train_model: Training pipeline with callbacks
    - evaluate_model: Evaluation and metrics calculation
    - predict: Single/batch prediction inference
    - main: Complete pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "EchoEmotion"
__description__ = "Speech Emotion Recognition using CNN+BiLSTM+Attention"

from .config import (
    EMOTIONS, NUM_CLASSES, EMOTION_TO_IDX, IDX_TO_EMOTION,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, DATA_DIR,
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH
)

from .data_processor import DataLoader, AudioProcessor, DataAugmentor
from .model_builder import build_ser_model, compile_model, MultiHeadAttention
from .train_model import train_model, evaluate_training
from .evaluate_model import ModelEvaluator, generate_evaluation_report
from .predict import SERPredictor, load_and_predict

__all__ = [
    # Config
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
    
    # Data processing
    'DataLoader',
    'AudioProcessor',
    'DataAugmentor',
    
    # Model building
    'build_ser_model',
    'compile_model',
    'MultiHeadAttention',
    
    # Training
    'train_model',
    'evaluate_training',
    
    # Evaluation
    'ModelEvaluator',
    'generate_evaluation_report',
    
    # Prediction
    'SERPredictor',
    'load_and_predict',
]
