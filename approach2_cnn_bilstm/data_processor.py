"""
Data Processing Module
Handles loading, preprocessing, and augmentation of audio data for SER
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from config import (
    EMOTIONS,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    TARGET_DURATION,
    MIN_DURATION,
    MAX_DURATION,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    USE_AUGMENTATION,
    AUGMENTATION_PROB,
    SPEC_AUGMENT_FREQ_MASK_PARAM,
    SPEC_AUGMENT_TIME_MASK_PARAM,
    SPEC_AUGMENT_NUM_MASKS,
    TIME_SHIFT_RANGE,
    NOISE_FACTOR,
    LOG_LEVEL,
    NUM_WORKERS,
    PIN_MEMORY,
)

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio loading and mel-spectrogram extraction."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = int(TARGET_DURATION * sample_rate // hop_length)

    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        try:
            y, _ = librosa.load(file_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=self.sample_rate)

            if duration < MIN_DURATION:
                logger.warning(f"Audio too short ({duration:.2f}s): {file_path}")
                return None

            if duration > MAX_DURATION:
                max_samples = int(MAX_DURATION * self.sample_rate)
                y = y[:max_samples]

            return y
        except Exception as exc:
            logger.error(f"Error loading audio {file_path}: {exc}")
            return None

    def extract_melspectrogram(self, y: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def pad_or_trim(self, mel_spec: np.ndarray) -> np.ndarray:
        current_length = mel_spec.shape[1]
        if current_length >= self.target_length:
            start_idx = (current_length - self.target_length) // 2
            mel_spec = mel_spec[:, start_idx:start_idx + self.target_length]
        else:
            pad_total = self.target_length - current_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            mel_spec = np.pad(mel_spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-80)
        return mel_spec

    def process_audio(self, file_path: str) -> Optional[np.ndarray]:
        y = self.load_audio(file_path)
        if y is None:
            return None
        mel_spec = self.extract_melspectrogram(y)
        return self.pad_or_trim(mel_spec)


class DataAugmentor:
    """Handles augmentation for spectrograms."""

    def spec_augment(self, mel_spec: np.ndarray) -> np.ndarray:
        mel_spec = mel_spec.copy()

        for _ in range(SPEC_AUGMENT_NUM_MASKS):
            f = np.random.randint(0, max(1, SPEC_AUGMENT_FREQ_MASK_PARAM))
            if f == 0 or mel_spec.shape[0] <= f:
                continue
            f_start = np.random.randint(0, mel_spec.shape[0] - f)
            mel_spec[f_start:f_start + f, :] = mel_spec.min()

        for _ in range(SPEC_AUGMENT_NUM_MASKS):
            t = np.random.randint(0, max(1, SPEC_AUGMENT_TIME_MASK_PARAM))
            if t == 0 or mel_spec.shape[1] <= t:
                continue
            t_start = np.random.randint(0, mel_spec.shape[1] - t)
            mel_spec[:, t_start:t_start + t] = mel_spec.min()

        return mel_spec

    def add_noise(self, mel_spec: np.ndarray) -> np.ndarray:
        return mel_spec + np.random.normal(0, NOISE_FACTOR, mel_spec.shape)

    def time_shift(self, mel_spec: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(TIME_SHIFT_RANGE[0], TIME_SHIFT_RANGE[1])
        shift_steps = int(shift * mel_spec.shape[1])
        return np.roll(mel_spec, shift=shift_steps, axis=1)

    def augment(self, mel_spec: np.ndarray) -> np.ndarray:
        if (not USE_AUGMENTATION) or (np.random.random() > AUGMENTATION_PROB):
            return mel_spec

        out = mel_spec
        if np.random.random() < 0.7:
            out = self.spec_augment(out)
        if np.random.random() < 0.5:
            out = self.add_noise(out)
        if np.random.random() < 0.3:
            out = self.time_shift(out)
        return out


class SpectrogramDataset(Dataset):
    """PyTorch dataset for mel-spectrograms."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augmentor: Optional[DataAugmentor] = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        mel = self.X[idx]
        if self.augmentor is not None:
            mel = self.augmentor.augment(mel)

        mel = np.expand_dims(mel, axis=0)  # (1, n_mels, time)
        return torch.from_numpy(mel).float(), torch.tensor(self.y[idx], dtype=torch.long)


class DataLoader:
    """Loads and manages training/validation/test datasets."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor()
        self.augmentor = DataAugmentor()
        self.scaler = StandardScaler()
        logger.info(f"Initializing DataLoader with data directory: {data_dir}")

    def load_dataset(self) -> Tuple[List[np.ndarray], List[int]]:
        all_spectrograms: List[np.ndarray] = []
        all_labels: List[int] = []

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
        original_shape = X.shape
        X_reshaped = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def _transform_with_scaler(self, X: np.ndarray) -> np.ndarray:
        X_reshaped = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(X.shape)

    def prepare_dataset(self, test_size: float = TEST_SPLIT, val_size: float = VAL_SPLIT) -> Dict:
        X, y = self.load_dataset()
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=RANDOM_SEED,
            stratify=y_train_val,
        )

        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")

        X_train_scaled = self.normalize_data(X_train)
        X_val_scaled = self._transform_with_scaler(X_val)
        X_test_scaled = self._transform_with_scaler(X_test)

        return {
            'X_train': X_train_scaled.astype(np.float32),
            'y_train': y_train,
            'X_val': X_val_scaled.astype(np.float32),
            'y_val': y_val,
            'X_test': X_test_scaled.astype(np.float32),
            'y_test': y_test,
            'scaler': self.scaler,
        }

    def create_torch_dataset(self, X: np.ndarray, y: np.ndarray, batch_size: int, augment: bool = False, shuffle: bool = True):
        dataset = SpectrogramDataset(X, y, augmentor=self.augmentor if augment else None)
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )


if __name__ == '__main__':
    from config import DATA_DIR, BATCH_SIZE

    loader = DataLoader(DATA_DIR)
    data = loader.prepare_dataset()

    print("\nDataset prepared successfully!")
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")

    train_ds = loader.create_torch_dataset(data['X_train'], data['y_train'], BATCH_SIZE, augment=True)
    val_ds = loader.create_torch_dataset(data['X_val'], data['y_val'], BATCH_SIZE, augment=False, shuffle=False)
    test_ds = loader.create_torch_dataset(data['X_test'], data['y_test'], BATCH_SIZE, augment=False, shuffle=False)

    print(f"Train dataset batches: {len(train_ds)}")
    print(f"Val dataset batches: {len(val_ds)}")
    print(f"Test dataset batches: {len(test_ds)}")
    print("Data loading test passed!")
