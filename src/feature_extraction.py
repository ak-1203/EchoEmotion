import librosa
import numpy as np


def extract_features(file_path, n_mfcc=13):
    """
    Extract audio features from a single file
    
    Returns:
        feature_vector (np.array): shape (58,)
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Add this after MFCC

        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
        delta_mfcc_std = np.std(delta_mfcc, axis=1)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # RMS
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_std = np.std(spec_cent)

        # Spectral Rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_roll_mean = np.mean(spec_roll)
        spec_roll_std = np.std(spec_roll)

        features = np.concatenate([
            mfcc_mean, mfcc_std,
            delta_mfcc_mean, delta_mfcc_std,
            [zcr_mean, zcr_std],
            [rms_mean, rms_std],
            chroma_mean, chroma_std,
            [spec_cent_mean, spec_cent_std],
            [spec_roll_mean, spec_roll_std]
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None