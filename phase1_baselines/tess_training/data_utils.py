from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import CLASS_TO_ID, CLASSES, DATA_DIR, SR
from feature_extraction import (
    extract_features,
    extract_mel_spectrogram,
    extract_mfcc_sequence,
    normalize_matrix,
    pad_or_trim_time_axis,
)


def parse_tess_label(folder_name):
    name = folder_name.lower().strip()

    if "_" in name:
        name = name.split("_", 1)[1]

    aliases = {
        "pleasant_surprise": "surprised",
        "pleasant_surprised": "surprised",
        "surprised": "surprised",
        "surprise": "surprised",
        "suprised": "surprised",
        "suprise": "surprised",
    }

    name = aliases.get(name, name)
    return name if name in CLASSES else None


def speaker_from_path(path):
    folder = Path(path).parent.name.upper().strip()
    if "_" in folder:
        return folder.split("_", 1)[0]
    return folder[:3]


def get_speaker_split_indices(paths, test_speaker="YAF"):
    train_idx, test_idx = [], []
    test_speaker = test_speaker.upper().strip()

    for i, p in enumerate(paths):
        speaker = speaker_from_path(p)
        if speaker == test_speaker:
            test_idx.append(i)
        else:
            train_idx.append(i)

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def load_tess_dataset(data_dir=DATA_DIR, sr=SR):
    samples = []
    wav_files = sorted(Path(data_dir).rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {data_dir}")

    for wav_path in wav_files:
        label = parse_tess_label(wav_path.parent.name)
        if label is None:
            continue

        try:
            audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
            if audio is None or len(audio) == 0:
                continue

            samples.append({
                "path": str(wav_path),
                "label": label,
                "y": CLASS_TO_ID[label],
                "speaker": speaker_from_path(wav_path),
                "audio": audio.astype(np.float32),
            })
        except Exception as exc:
            print(f"Skipping {wav_path.name}: {exc}")

    if not samples:
        raise RuntimeError("No valid TESS samples were loaded.")

    le = LabelEncoder()
    le.fit(CLASSES)

    max_mel_len = 0
    max_seq_len = 0

    for sample in samples:
        mel = extract_mel_spectrogram(sample["audio"], sr)
        seq = extract_mfcc_sequence(sample["audio"], sr)
        max_mel_len = max(max_mel_len, mel.shape[1])
        max_seq_len = max(max_seq_len, seq.shape[0])

    return samples, le, max_mel_len, max_seq_len


def augment_audio(y, sr, rng):
    y_aug = y.astype(np.float32).copy()

    if rng.random() < 0.5 and len(y_aug) > 16:
        rate = float(rng.uniform(0.90, 1.10))
        y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

    if rng.random() < 0.5 and len(y_aug) > 16:
        n_steps = float(rng.uniform(-2.0, 2.0))
        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)

    if rng.random() < 0.5:
        noise_scale = float(rng.uniform(0.0015, 0.004))
        noise = rng.normal(0.0, noise_scale, size=y_aug.shape).astype(np.float32)
        y_aug = y_aug + noise

    if rng.random() < 0.3 and len(y_aug) > 16:
        max_shift = max(1, int(0.1 * len(y_aug)))
        shift = int(rng.integers(-max_shift, max_shift + 1))
        y_aug = np.roll(y_aug, shift)

    return np.clip(y_aug, -1.0, 1.0)


def build_ml_dataset(samples, indices, sr=SR):
    X, y = [], []

    for idx in indices:
        sample = samples[idx]
        X.append(extract_features(sample["audio"], sr))
        y.append(sample["y"])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def build_deep_dataset(
    samples,
    indices,
    max_mel_len,
    max_seq_len,
    sr=SR,
    augment=False,
    augment_factor=1,
    augment_happy_factor=2,
    seed=42,
):
    X_mel, X_seq, y = [], [], []
    rng = np.random.default_rng(seed)

    for idx in indices:
        sample = samples[idx]
        label = sample["label"]

        repeats = 1
        if augment:
            repeats += augment_factor
            if label == "happy":
                repeats += augment_happy_factor

        for rep in range(repeats):
            audio = sample["audio"]
            if augment and rep > 0:
                audio = augment_audio(audio, sr, rng)

            mel = extract_mel_spectrogram(audio, sr)
            mel = normalize_matrix(mel)
            mel = pad_or_trim_time_axis(mel, max_mel_len, axis=1)

            seq = extract_mfcc_sequence(audio, sr)
            seq = normalize_matrix(seq)
            seq = pad_or_trim_time_axis(seq, max_seq_len, axis=0)

            X_mel.append(mel[..., np.newaxis])
            X_seq.append(seq)
            y.append(sample["y"])

    return (
        np.asarray(X_mel, dtype=np.float32),
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
    )


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )