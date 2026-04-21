"""
Data Processing Module
Handles loading, preprocessing, and augmentation of audio data for SER
"""

import os
import logging
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import (
    EMOTIONS, EMOTION_TO_IDX, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    TARGET_DURATION, MIN_DURATION, MAX_DURATION, TRAIN_SPLIT, VAL_SPLIT,
    TEST_SPLIT, RANDOM_SEED, USE_AUGMENTATION, AUGMENTATION_PROB,
    SPEC_AUGMENT_FREQ_MASK_PARAM, SPEC_AUGMENT_TIME_MASK_PARAM,
    SPEC_AUGMENT_NUM_MASKS, PITCH_SHIFT_RANGE, TIME_SHIFT_RANGE, NOISE_FACTOR,
    LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio loading and mel-spectrogram extraction"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS,
                 n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate (Hz)
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Number of samples between frames
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = int(TARGET_DURATION * sample_rate // hop_length)
        
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Load audio file and resample to target sample rate"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Check duration
            duration = librosa.get_duration(y=y, sr=self.sample_rate)
            if duration < MIN_DURATION:
                logger.warning(f"Audio too short ({duration:.2f}s): {file_path}")
                return None
            
            if duration > MAX_DURATION:
                # Trim to max duration
                max_samples = int(MAX_DURATION * self.sample_rate)
                y = y[:max_samples]
            
            return y
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            return None
    
    def extract_melspectrogram(self, y: np.ndarray, power: float = 2.0) -> np.ndarray:
        """
        Extract mel-spectrogram from audio
        
        Args:
            y: Audio time series
            power: Power of the spectrogram (2.0 for energy)
            
        Returns:
            Mel-spectrogram (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=power
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def pad_or_trim(self, mel_spec: np.ndarray) -> np.ndarray:
        """Pad or trim mel-spectrogram to target length"""
        current_length = mel_spec.shape[1]
        
        if current_length >= self.target_length:
            # Trim from center
            start_idx = (current_length - self.target_length) // 2
            mel_spec = mel_spec[:, start_idx:start_idx + self.target_length]
        else:
            # Pad symmetrically
            pad_total = self.target_length - current_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            mel_spec = np.pad(mel_spec, ((0, 0), (pad_left, pad_right)), 
                            mode='constant', constant_values=-80)
        
        return mel_spec
    
    def process_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Complete audio processing pipeline"""
        y = self.load_audio(file_path)
        if y is None:
            return None
        
        mel_spec = self.extract_melspectrogram(y)
        mel_spec = self.pad_or_trim(mel_spec)
        
        return mel_spec


class DataAugmentor:
    """Handles data augmentation for audio spectrograms"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH):
        """Initialize augmentor"""
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def spec_augment(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to mel-spectrogram
        
        Args:
            mel_spec: Input mel-spectrogram (n_mels, time_steps)
            
        Returns:
            Augmented mel-spectrogram
        """
        mel_spec = mel_spec.copy()
        
        # Frequency masking
        for _ in range(SPEC_AUGMENT_NUM_MASKS):
            f = np.random.randint(0, SPEC_AUGMENT_FREQ_MASK_PARAM)
            f_start = np.random.randint(0, mel_spec.shape[0] - f)
            mel_spec[f_start:f_start + f, :] = mel_spec.min()
        
        # Time masking
        for _ in range(SPEC_AUGMENT_NUM_MASKS):
            t = np.random.randint(0, SPEC_AUGMENT_TIME_MASK_PARAM)
            t_start = np.random.randint(0, mel_spec.shape[1] - t)
            mel_spec[:, t_start:t_start + t] = mel_spec.min()
        
        return mel_spec
    
    def add_noise(self, mel_spec: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to spectrogram"""
        noise = np.random.normal(0, NOISE_FACTOR, mel_spec.shape)
        return mel_spec + noise
    
    def time_shift(self, mel_spec: np.ndarray) -> np.ndarray:
        """Apply time shifting"""
        shift = np.random.uniform(TIME_SHIFT_RANGE[0], TIME_SHIFT_RANGE[1])
        shift_samples = int(shift * mel_spec.shape[1])
        
        if shift_samples > 0:
            mel_spec = np.concatenate([mel_spec[:, shift_samples:], 
                                       mel_spec[:, :shift_samples]], axis=1)
        elif shift_samples < 0:
            mel_spec = np.concatenate([mel_spec[:, :shift_samples], 
                                       mel_spec[:, -shift_samples:]], axis=1)
        
        return mel_spec
    
    def augment(self, mel_spec: np.ndarray) -> np.ndarray:
        """Apply random augmentations"""
        if not USE_AUGMENTATION or np.random.random() > AUGMENTATION_PROB:
            return mel_spec
        
        augmentations = [
            self.spec_augment,
            self.add_noise,
            self.time_shift
        ]
        
        # Randomly apply augmentations
        if np.random.random() < 0.7:
            mel_spec = self.spec_augment(mel_spec)
        if np.random.random() < 0.5:
            mel_spec = self.add_noise(mel_spec)
        if np.random.random() < 0.3:
            mel_spec = self.time_shift(mel_spec)
        
        return mel_spec


class DataLoader:
    """Loads and manages training/validation/test datasets"""
    
    def __init__(self, data_dir: Path):
        """Initialize data loader"""
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor()
        self.augmentor = DataAugmentor()
        self.scaler = StandardScaler()
        
        logger.info(f"Initializing DataLoader with data directory: {data_dir}")
    
    def load_dataset(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load all audio files and extract mel-spectrograms
        
        Returns:
            Tuple of (mel_spectrograms, labels)
        """
        all_spectrograms = []
        all_labels = []
        
        logger.info("Loading dataset...")
        
        for emotion_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = self.data_dir / emotion
            
            if not emotion_dir.exists():
                logger.warning(f"Emotion directory not found: {emotion_dir}")
                continue
            
            audio_files = list(emotion_dir.glob('*.wav'))
            logger.info(f"Found {len(audio_files)} files for emotion: {emotion}")
            
            for file_path in audio_files:
                mel_spec = self.audio_processor.process_audio(str(file_path))
                
                if mel_spec is not None:
                    all_spectrograms.append(mel_spec)
                    all_labels.append(emotion_idx)
        
        logger.info(f"Loaded {len(all_spectrograms)} spectrograms from {len(EMOTIONS)} emotion classes")
        
        return all_spectrograms, all_labels
    
    def normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalization of spectrograms"""
        # Reshape for scaling: (samples, features)
        original_shape = X.shape
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Fit scaler on training data and transform
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def prepare_dataset(self, test_size: float = TEST_SPLIT, 
                       val_size: float = VAL_SPLIT) -> Dict:
        """
        Prepare train/validation/test split
        
        Returns:
            Dictionary with train, val, test splits
        """
        X, y = self.load_dataset()
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # First split: train+val vs test (stratified)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        # Second split: train vs val (stratified)
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted,
            random_state=RANDOM_SEED, stratify=y_train_val
        )
        
        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        # Normalize (fit on train, apply to all)
        X_train_scaled = self.normalize_data(X_train)
        X_val_scaled = (X_val - self.scaler.mean_) / np.sqrt(self.scaler.var_)
        X_test_scaled = (X_test - self.scaler.mean_) / np.sqrt(self.scaler.var_)
        
        return {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'scaler': self.scaler
        }
    
    def create_tf_dataset(self, X: np.ndarray, y: np.ndarray, 
                         batch_size: int, augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with augmentation
        
        Args:
            X: Spectrograms
            y: Labels
            batch_size: Batch size
            augment: Whether to apply augmentation
            
        Returns:
            TF Dataset
        """
        def augment_fn(mel_spec, label):
            mel_spec = tf.numpy_function(
                self.augmentor.augment,
                [mel_spec],
                tf.float32
            )
            return mel_spec, label
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if augment:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.shuffle(buffer_size=min(1000, len(X)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


if __name__ == '__main__':
    # Test data loading
    from config import DATA_DIR, BATCH_SIZE
    
    loader = DataLoader(DATA_DIR)
    data = loader.prepare_dataset()
    
    print(f"\nDataset prepared successfully!")
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    
    # Create TF datasets
    train_ds = loader.create_tf_dataset(data['X_train'], data['y_train'], BATCH_SIZE, augment=True)
    val_ds = loader.create_tf_dataset(data['X_val'], data['y_val'], BATCH_SIZE)
    test_ds = loader.create_tf_dataset(data['X_test'], data['y_test'], BATCH_SIZE)
    
    print(f"Train dataset batches: {len(list(train_ds))}")
    print("Data loading test passed!")
