import numpy as np
import librosa

from config import N_MELS, N_MFCC


def extract_features(y, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        np.mean(delta2, axis=1),
        np.std(delta2, axis=1),
        np.array([np.mean(zcr), np.std(zcr)]),
        np.array([np.mean(rms), np.std(rms)]),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.array([np.mean(centroid), np.std(centroid)]),
        np.array([np.mean(rolloff), np.std(rolloff)]),
    ])

    return features.astype(np.float32)


def extract_mel_spectrogram(y, sr, n_mels=N_MELS, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def extract_mfcc_sequence(y, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)
    return stacked.T.astype(np.float32)


def pad_or_trim_time_axis(arr, target_len, axis=1):
    if arr.shape[axis] == target_len:
        return arr

    if arr.shape[axis] > target_len:
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, target_len)
        return arr[tuple(slicer)]

    pad_width = target_len - arr.shape[axis]
    pad_config = [(0, 0)] * arr.ndim
    pad_config[axis] = (0, pad_width)
    return np.pad(arr, pad_config, mode="constant", constant_values=0.0)


def normalize_matrix(matrix):
    mean = float(matrix.mean())
    std = float(matrix.std())
    return ((matrix - mean) / (std + 1e-8)).astype(np.float32)