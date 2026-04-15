import os
import numpy as np
from tqdm import tqdm

from src.feature_extraction import extract_features


# Emotion mapping
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def get_emotion_from_filename(filename):
    """
    Extract emotion label from RAVDESS filename
    """
    parts = filename.split('-')
    emotion_id = parts[2]
    return EMOTION_MAP[emotion_id]


def load_dataset(data_path):
    """
    Load dataset and extract features
    
    Returns:
        X (np.array): feature matrix
        y (np.array): labels
    """
    X = []
    y = []

    audio_files = []

    # Walk through all folders
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    print(f"Found {len(audio_files)} audio files")

    # Extract features
    for file_path in tqdm(audio_files):
        features = extract_features(file_path)

        if features is not None:
            X.append(features)

            filename = os.path.basename(file_path)
            emotion = get_emotion_from_filename(filename)
            y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y