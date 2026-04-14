# Speech Emotion Recognition - Detailed Implementation Roadmap

## PROJECT OVERVIEW

**Goal:** Build a production-grade, publication-ready Speech Emotion Recognition system

**Duration:** 10 weeks (part-time expected work)

**Target Accuracies:**
- Week 2: 72-85% (classical ML baseline)
- Week 4: 92-97% (deep learning)
- Week 6: 95%+ (advanced method)
- Week 8: 80-90% (Hindi language)
- Week 10: 80-95% (custom test set)

**Final Deliverables:**
- ✅ Trained models (SVM, CNN+BiLSTM, Advanced)
- ✅ Cross-dataset evaluation results
- ✅ Custom Hindi dataset (optional)
- ✅ Custom test dataset with results
- ✅ GitHub repository (well-documented)
- ✅ Project report (5-10 pages)
- ✅ Presentation slides

---

## WEEK 1: SETUP & DATA PREPARATION

### Learning Objectives
- Understand audio signal processing basics
- Learn to load and manipulate audio files
- Understand MFCC and spectrograms
- Set up development environment

### Tasks

#### Task 1.1: Environment Setup (2 hours)
```bash
# Create virtual environment
python -m venv ser_env
source ser_env/bin/activate  # On Windows: ser_env\Scripts\activate

# Install required packages
pip install librosa numpy pandas matplotlib seaborn jupyter
pip install scikit-learn scipy soundfile
pip install tensorflow keras
pip install torch torchaudio  # Optional, if using PyTorch

# Create project structure
mkdir SER-Project
cd SER-Project
mkdir data notebooks src results
```

#### Task 1.2: Download RAVDESS Dataset (2 hours)
```python
# Download manually from: https://zenodo.org/record/1188976
# Extract to: data/ravdess/

# Verify structure
import os
ravdess_path = 'data/ravdess/'
audio_files = [f for f in os.listdir(ravdess_path) if f.endswith('.wav')]
print(f"Total files: {len(audio_files)}")  # Should be ~1440
```

#### Task 1.3: Audio Loading & Exploration (3 hours)
```python
# Create: notebooks/01_eda_data_exploration.ipynb

import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load sample audio
audio_file = 'data/ravdess/03-01-01-01-02-01-01.wav'
y, sr = librosa.load(audio_file, sr=16000)

print(f"Duration: {len(y)/sr:.2f} seconds")
print(f"Sampling rate: {sr} Hz")
print(f"Audio shape: {y.shape}")

# Visualize
plt.figure(figsize=(12, 4))

# Waveform
plt.subplot(1, 2, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')

# Spectrogram
plt.subplot(1, 2, 2)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram')

plt.tight_layout()
plt.savefig('results/sample_audio_exploration.png', dpi=150)
plt.show()

# Emotion mapping from RAVDESS filename
# Format: 03-01-01-01-02-01-01.wav
# Position 3: Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Count emotions
from collections import Counter
emotion_counts = Counter()
for f in audio_files:
    emotion_id = f.split('-')[2]
    emotion_counts[emotions[emotion_id]] += 1

print("Emotion distribution:")
for emotion, count in emotion_counts.most_common():
    print(f"  {emotion}: {count}")
```

#### Task 1.4: Feature Extraction Exploration (3 hours)
```python
# Create: src/feature_extraction.py

import librosa
import numpy as np

def extract_features(y, sr, n_mfcc=13):
    """
    Extract hand-crafted features from audio signal
    
    Returns: 1D feature vector
    """
    
    # 1. MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)  # (13,)
    mfcc_std = np.std(mfcc, axis=1)    # (13,)
    
    # 2. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 3. Root Mean Square (Energy)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # 4. Chroma Features (12 coefficients)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # (12,)
    chroma_std = np.std(chroma, axis=1)    # (12,)
    
    # 5. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_std = np.std(spectral_centroid)
    
    # 6. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_std = np.std(spectral_rolloff)
    
    # Combine all features
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [zcr_mean, zcr_std],
        [rms_mean, rms_std],
        chroma_mean, chroma_std,
        [spectral_centroid_mean, spectral_centroid_std],
        [spectral_rolloff_mean, spectral_rolloff_std]
    ])
    
    return features  # Total: 13+13+2+2+12+12+2+2 = 58 features


def extract_mel_spectrogram(y, sr, n_mels=128):
    """
    Extract mel-spectrogram (for deep learning)
    
    Returns: 2D feature matrix (n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


# Test
if __name__ == '__main__':
    y, sr = librosa.load('data/ravdess/03-01-01-01-02-01-01.wav')
    
    features = extract_features(y, sr)
    print(f"Features shape: {features.shape}")  # (58,)
    
    mel_spec = extract_mel_spectrogram(y, sr)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")  # (128, ~130)
```

### Deliverables for Week 1
- [ ] Environment set up with all packages
- [ ] RAVDESS dataset downloaded and explored
- [ ] Data exploration notebook (EDA) with visualizations
- [ ] Feature extraction functions written and tested
- [ ] Emotion distribution analyzed
- [ ] Project structure created on GitHub (empty repo)

### Time Budget: ~14 hours

---

## WEEK 2: CLASSICAL ML BASELINE

### Learning Objectives
- Implement SVM, Random Forest, XGBoost classifiers
- Understand feature scaling and preprocessing
- Perform hyperparameter tuning
- Achieve 72-85% accuracy baseline

### Tasks

#### Task 2.1: Data Loading & Preprocessing (3 hours)
```python
# Create: notebooks/02_data_preprocessing.ipynb

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.feature_extraction import extract_features

def load_dataset_features(data_path, n_mfcc=13):
    """Load all audio files and extract features"""
    
    features_list = []
    labels_list = []
    
    emotions_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    audio_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    
    for audio_file in audio_files:
        try:
            # Load audio
            y, sr = librosa.load(os.path.join(data_path, audio_file), sr=16000)
            
            # Extract features
            features = extract_features(y, sr, n_mfcc=n_mfcc)
            features_list.append(features)
            
            # Get emotion label
            emotion_id = audio_file.split('-')[2]
            emotion = emotions_map[emotion_id]
            labels_list.append(emotion)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y

# Load data
X, y = load_dataset_features('data/ravdess/')

print(f"Features shape: {X.shape}")  # (1440, 58)
print(f"Labels shape: {y.shape}")    # (1440,)

# Check for missing values
print(f"NaN values: {np.isnan(X).sum()}")

# Emotion distribution
from collections import Counter
emotion_dist = Counter(y)
print("\nEmotion distribution:")
for emotion, count in emotion_dist.most_common():
    print(f"  {emotion}: {count}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled data mean: {X_scaled.mean(axis=0)[:5]}")  # Should be ~0
print(f"Scaled data std: {X_scaled.std(axis=0)[:5]}")    # Should be ~1

# Save for later use
np.save('data/X_features.npy', X)
np.save('data/y_labels.npy', y)
```

#### Task 2.2: Train-Test Split & Baseline Models (4 hours)
```python
# Continue in notebook: 02_data_preprocessing.ipynb

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
X = np.load('data/X_features.npy')
y = np.load('data/y_labels.npy')

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Model 1: SVM
print("\n=== SVM Classifier ===")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"Accuracy: {svm_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, svm_pred))

# Model 2: Random Forest
print("\n=== Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Model 3: Logistic Regression
print("\n=== Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_accuracy:.4f}")

# Feature importance (RF only)
feature_importance = rf_model.feature_importances_
top_indices = np.argsort(feature_importance)[-10:]
print(f"\nTop 10 important features (Random Forest):")
for idx in reversed(top_indices):
    print(f"  Feature {idx}: {feature_importance[idx]:.4f}")

# Visualize confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, model, pred, name in zip(axes, 
                                  [svm_model, rf_model, lr_model],
                                  [svm_pred, rf_pred, lr_pred],
                                  ['SVM', 'Random Forest', 'Logistic Regression']):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=svm_model.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, pred):.2%}')

plt.tight_layout()
plt.savefig('results/classical_ml_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary
results_df = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [svm_accuracy, rf_accuracy, lr_accuracy]
})
print("\n=== SUMMARY ===")
print(results_df.to_string(index=False))
results_df.to_csv('results/classical_ml_results.csv', index=False)
```

#### Task 2.3: Hyperparameter Tuning (3 hours)
```python
# Create: src/hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV

def tune_svm(X_train, y_train):
    """Tune SVM hyperparameters"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# In notebook: tune and save best models
best_svm = tune_svm(X_train, y_train)
best_svm_pred = best_svm.predict(X_test)
print(f"Tuned SVM accuracy: {accuracy_score(y_test, best_svm_pred):.4f}")

best_rf = tune_random_forest(X_train, y_train)
best_rf_pred = best_rf.predict(X_test)
print(f"Tuned RF accuracy: {accuracy_score(y_test, best_rf_pred):.4f}")
```

#### Task 2.4: Analysis & Results Summary (2 hours)
```python
# Prepare summary report
import matplotlib.pyplot as plt

# Create results summary
baseline_results = {
    'Method': ['SVM', 'Random Forest', 'Logistic Regression', 'Tuned SVM', 'Tuned RF'],
    'Accuracy': [0.75, 0.78, 0.72, 0.80, 0.82],  # Example values
    'Precision': [0.76, 0.79, 0.73, 0.81, 0.83],
    'Recall': [0.74, 0.76, 0.70, 0.79, 0.81]
}

results_df = pd.DataFrame(baseline_results)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
results_df.set_index('Method')[['Accuracy', 'Precision', 'Recall']].plot(
    kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c']
)
ax.set_ylabel('Score')
ax.set_title('Classical ML Baseline Comparison')
ax.set_ylim([0.6, 1.0])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/baseline_comparison.png', dpi=150, bbox_inches='tight')

print("\n=== WEEK 2 RESULTS ===")
print(results_df.to_string(index=False))
print(f"\nBest Model: {results_df.loc[results_df['Accuracy'].idxmax(), 'Method']}")
print(f"Best Accuracy: {results_df['Accuracy'].max():.2%}")

# Save best model
import pickle
with open('results/best_classical_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)  # Save best model
```

### Deliverables for Week 2
- [ ] Dataset loaded with 1440 samples
- [ ] Features extracted (58-dimensional vectors)
- [ ] Train-test split created (80-20)
- [ ] 5 models trained: SVM, RF, LR, Tuned SVM, Tuned RF
- [ ] All confusion matrices visualized
- [ ] Results comparison table (Accuracy, Precision, Recall, F1)
- [ ] Baseline report (best: 78-82% accuracy)
- [ ] Feature importance analysis (top 10 features identified)
- [ ] GitHub commit: "Week 2: Classical ML Baseline"

### Time Budget: ~15 hours

### Expected Performance
- SVM: 75-78%
- Random Forest: 78-82% (best)
- Logistic Regression: 70-75%
- After tuning: +2-4% improvement

---

## WEEK 3-4: DEEP LEARNING - CNN+BiLSTM

### Learning Objectives
- Build CNN+BiLSTM hybrid architecture
- Implement data augmentation
- Train end-to-end deep learning model
- Achieve 92-97% accuracy

### Tasks

#### Task 3.1: Prepare Mel-Spectrogram Data (3 hours)
```python
# Create: notebooks/03_prepare_deep_learning_data.ipynb

import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def create_mel_spectrogram_dataset(data_path, n_mels=128, sr=16000, 
                                    n_fft=2048, hop_length=512):
    """
    Create dataset of mel-spectrograms for deep learning
    """
    
    X_list = []
    y_list = []
    
    emotions_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    emotion_to_id = {v: k for k, v in emotions_map.items()}
    
    audio_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    
    max_len = 0  # Track max length
    
    for audio_file in audio_files:
        try:
            # Load audio
            y, sr = librosa.load(os.path.join(data_path, audio_file), sr=sr)
            
            # Create mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, 
                n_fft=n_fft, hop_length=hop_length
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Get emotion
            emotion_id = audio_file.split('-')[2]
            emotion = emotions_map[emotion_id]
            emotion_num = int(emotion_id)
            
            X_list.append(log_mel_spec)
            y_list.append(emotion_num - 1)  # 0-indexed
            
            max_len = max(max_len, log_mel_spec.shape[1])
            
        except Exception as e:
            print(f"Error: {audio_file}: {e}")
    
    print(f"Max spectrogram length: {max_len}")
    return X_list, y_list, max_len

# Create dataset
X_spectrograms, y_labels, max_len = create_mel_spectrogram_dataset('data/ravdess/')
print(f"Dataset size: {len(X_spectrograms)} samples")
print(f"Max length: {max_len}")  # ~130 for 3-second audio

# Pad all spectrograms to same length
def pad_spectrograms(X_list, max_len):
    """Pad all spectrograms to same length"""
    X_padded = []
    for spec in X_list:
        if spec.shape[1] < max_len:
            pad_width = max_len - spec.shape[1]
            spec_padded = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec_padded = spec[:, :max_len]
        X_padded.append(spec_padded)
    
    return np.array(X_padded)

X_padded = pad_spectrograms(X_spectrograms, max_len)
y_array = np.array(y_labels)

print(f"Padded shape: {X_padded.shape}")  # (1440, 128, ~130)

# Normalize
X_normalized = np.zeros_like(X_padded)
for i in range(X_padded.shape[0]):
    mean = X_padded[i].mean()
    std = X_padded[i].std()
    if std > 0:
        X_normalized[i] = (X_padded[i] - mean) / std
    else:
        X_normalized[i] = X_padded[i] - mean

# Save
np.save('data/X_mel_specs.npy', X_normalized)
np.save('data/y_mel_labels.npy', y_array)

print("Data saved!")
```

#### Task 3.2: Data Augmentation (3 hours)
```python
# Create: src/data_augmentation.py

import librosa
import numpy as np

def augment_audio(y, sr, num_augmentations=3):
    """
    Apply multiple augmentation techniques to audio
    """
    augmented = []
    
    # Original
    augmented.append(y)
    
    # 1. Add Gaussian noise
    if num_augmentations >= 1:
        noise = np.random.normal(0, 0.005, len(y))
        y_noise = y + noise
        augmented.append(y_noise)
    
    # 2. Pitch shift
    if num_augmentations >= 2:
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented.append(y_pitch)
    
    # 3. Time stretch
    if num_augmentations >= 3:
        y_stretch = librosa.effects.time_stretch(y, rate=0.9)
        augmented.append(y_stretch)
    
    return augmented

def augment_spectrogram(mel_spec, num_augmentations=3):
    """
    Augmentation on spectrogram level
    """
    augmented = []
    
    # Original
    augmented.append(mel_spec)
    
    # 1. Add noise to spectrogram
    if num_augmentations >= 1:
        noise = np.random.normal(0, 0.01, mel_spec.shape)
        augmented.append(mel_spec + noise)
    
    # 2. Frequency shifting
    if num_augmentations >= 2:
        shift = np.random.randint(-5, 5)
        mel_shift = np.roll(mel_spec, shift, axis=0)
        augmented.append(mel_shift)
    
    # 3. Time shifting
    if num_augmentations >= 3:
        shift = np.random.randint(-10, 10)
        time_shift = np.roll(mel_spec, shift, axis=1)
        augmented.append(time_shift)
    
    return augmented

# In notebook: Apply augmentation
X_mel = np.load('data/X_mel_specs.npy')
y_mel = np.load('data/y_mel_labels.npy')

# Apply augmentation during training (using TensorFlow/Keras)
# This will be done inside the training loop
print("Augmentation functions ready for training!")
```

#### Task 3.3: Build CNN+BiLSTM Model (4 hours)
```python
# Create: src/models.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, LSTM, Bidirectional, 
    Dense, Dropout, Flatten, LayerNormalization, 
    GlobalAveragePooling2D, Attention
)

def create_cnn_bilstm_model(input_shape, num_classes, use_attention=False):
    """
    CNN + BiLSTM hybrid model for emotion recognition
    
    Input shape: (height=128, width=time_steps, channels=1)
    """
    model = Sequential([
        # CNN Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', 
               input_shape=input_shape, name='conv1'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),
        
        # CNN Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),
        
        # CNN Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),
        
        # Reshape for LSTM
        # From (batch, height, width, channels) to (batch, time_steps, features)
        layers.Reshape((-1, 256)),  # (batch, time_steps, 256)
        
        # BiLSTM
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3), 
                     name='bilstm1'),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.3), 
                     name='bilstm2'),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # Output layer
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

def create_cnn_model(input_shape, num_classes):
    """Simpler CNN-only model"""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_lstm_model(input_shape, num_classes):
    """LSTM-only model"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Test models
if __name__ == '__main__':
    input_shape = (128, 130, 1)  # (freq_bins, time_steps, channels)
    num_classes = 8  # 8 emotions
    
    model_cnn_lstm = create_cnn_bilstm_model(input_shape, num_classes)
    model_cnn_lstm.summary()
    
    print(f"Model parameters: {model_cnn_lstm.count_params():,}")
```

#### Task 3.4: Training with Validation (4 hours)
```python
# Create: notebooks/04_train_deep_learning.ipynb

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.models import create_cnn_bilstm_model

# Load data
X = np.load('data/X_mel_specs.npy')
y = np.load('data/y_mel_labels.npy')

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Add channel dimension
X = X[:, :, :, np.newaxis]  # (1440, 128, 130, 1)
print(f"Input shape for model: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split train into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Build model
model = create_cnn_bilstm_model(
    input_shape=(128, 130, 1), 
    num_classes=8
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'results/best_cnn_lstm_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
from sklearn.metrics import classification_report, confusion_matrix
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=emotions))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Visualize training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualize confusion matrix
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emotions, yticklabels=emotions)
plt.title(f'CNN+BiLSTM Confusion Matrix\nAccuracy: {test_accuracy:.2%}')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/cnn_lstm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Save model
model.save('results/cnn_lstm_model_final.h5')
print("Model saved!")
```

### Deliverables for Week 3-4
- [ ] Mel-spectrogram dataset created (1440 samples, 128x130x1)
- [ ] Data augmentation functions implemented
- [ ] CNN+BiLSTM model built and compiled
- [ ] Model trained with early stopping
- [ ] Test accuracy: 92-97%
- [ ] Confusion matrices and classification reports
- [ ] Training history plots (accuracy and loss curves)
- [ ] Model checkpoints saved
- [ ] GitHub commit: "Week 4: CNN+BiLSTM Model - 95% accuracy"

### Time Budget: ~18 hours

### Expected Performance
- CNN+BiLSTM: 92-97% accuracy
- Per-emotion accuracy varies (happy/calm easier, fear/disgust harder)
- Training converges in 30-50 epochs with early stopping

---

## WEEKS 5-6: ADVANCED METHOD (Choose One)

### Option A: Ensemble + Attention + Explainability (Recommended for Publication)

#### Task 5.1: Build Ensemble (4 hours)
```python
# Create: src/ensemble_models.py

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

def create_ensemble_model(input_shape, num_classes, include_attention=True):
    """
    Ensemble of CNN, LSTM, and GRU models
    """
    
    # Input
    input_layer = layers.Input(shape=input_shape)
    
    # Branch 1: CNN
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)
    x1 = layers.Dense(128, activation='relu')(x1)
    branch1 = layers.Dropout(0.5)(x1)
    
    # Branch 2: Reshape + BiLSTM
    x2 = layers.Reshape((-1, 64))(input_layer)  # Reshape for LSTM
    x2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x2)
    x2 = layers.Bidirectional(layers.LSTM(64))(x2)
    branch2 = layers.Dropout(0.5)(x2)
    
    # Branch 3: Reshape + BiGRU
    x3 = layers.Reshape((-1, 64))(input_layer)
    x3 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x3)
    x3 = layers.Bidirectional(layers.GRU(64))(x3)
    branch3 = layers.Dropout(0.5)(x3)
    
    # Fusion with Attention
    if include_attention:
        # Concatenate all branches
        concatenated = layers.Concatenate()([branch1, branch2, branch3])
        
        # Attention mechanism
        attention_layer = layers.Dense(384, activation='relu')(concatenated)
        attention_weights = layers.Dense(384, activation='softmax')(attention_layer)
        fused = layers.Multiply()([concatenated, attention_weights])
    else:
        # Simple concatenation
        fused = layers.Concatenate()([branch1, branch2, branch3])
    
    # Classification head
    x = layers.Dense(256, activation='relu')(fused)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

# Test
if __name__ == '__main__':
    input_shape = (128, 130, 1)
    model = create_ensemble_model(input_shape, 8, include_attention=True)
    model.summary()
```

#### Task 5.2: Add Explainability (3 hours)
```python
# Create: src/explainability.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def get_grad_cam_heatmap(model, img_array, layer_name):
    """
    Generate Grad-CAM heatmap for a given image
    """
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def visualize_grad_cam(image, heatmap, emotion, save_path=None):
    """
    Visualize original spectrogram with Grad-CAM heatmap
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original spectrogram
    im1 = axes[0].imshow(image[0, :, :, 0], cmap='viridis', aspect='auto')
    axes[0].set_title('Original Mel-Spectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency')
    plt.colorbar(im1, ax=axes[0])
    
    # Heatmap
    im2 = axes[1].imshow(heatmap, cmap='hot', aspect='auto')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency')
    plt.colorbar(im2, ax=axes[1])
    
    # Overlay
    axes[2].imshow(image[0, :, :, 0], cmap='viridis', aspect='auto', alpha=0.6)
    axes[2].imshow(heatmap, cmap='hot', aspect='auto', alpha=0.4)
    axes[2].set_title(f'Overlay\nEmotion: {emotion}')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Test in notebook
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Get sample from test set
sample_idx = 10
sample_image = X_test[sample_idx:sample_idx+1]
sample_label = emotions[y_test[sample_idx]]

# Generate Grad-CAM
heatmap = get_grad_cam_heatmap(model, sample_image, 'conv3')
visualize_grad_cam(sample_image, heatmap, sample_label, 
                   save_path=f'results/gradcam_{sample_label}.png')

print("Grad-CAM visualizations generated!")
```

#### Task 5.3: Train Ensemble & Compare (4 hours)
```python
# In notebook: 05_train_advanced_methods.ipynb

# Build and train ensemble
ensemble_model = create_ensemble_model((128, 130, 1), 8, include_attention=True)

ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

ensemble_history = ensemble_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Evaluate
ensemble_accuracy = ensemble_model.evaluate(X_test, y_test)[1]
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Compare all models
comparison_df = pd.DataFrame({
    'Model': ['Classical ML (RF)', 'CNN Only', 'LSTM Only', 'CNN+BiLSTM', 'Ensemble+Attention'],
    'Accuracy': [0.82, 0.88, 0.90, 0.95, 0.96],  # Example values
    'Interpretability': ['Good', 'Medium', 'Medium', 'Medium', 'Excellent']
})

print("\n=== MODEL COMPARISON ===")
print(comparison_df.to_string(index=False))

# Save comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
models = comparison_df['Model']
accuracies = comparison_df['Accuracy']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
ax.barh(models, accuracies, color=colors)
ax.set_xlabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xlim([0.75, 1.0])
for i, v in enumerate(accuracies):
    ax.text(v - 0.02, i, f'{v:.1%}', va='center', ha='right', color='white', fontweight='bold')
plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Deliverables for Weeks 5-6

Choose **ONE** advanced method:
- **Option A (Ensemble+Attention)**: [ ] Ensemble model trained, [ ] Grad-CAM visualizations, [ ] 95-98% accuracy, [ ] Explainability analysis
- **Option B (Transfer Learning)**: [ ] ResNet50 pre-trained model fine-tuned, [ ] 93-96% accuracy, [ ] Fast training
- **Option C (Transformer)**: [ ] Vision Transformer model, [ ] 94-97% accuracy, [ ] Zero-shot transfer capability

Plus:
- [ ] Model comparison table (all 5 approaches)
- [ ] Visualization of advanced technique
- [ ] GitHub commit: "Week 6: Advanced Method - [Name] - [Accuracy]%"

---

## WEEKS 7-8: CROSS-DATASET & HINDI EXTENSION

### Task 7.1: Cross-Dataset Evaluation (3 hours)
```python
# Notebook: 06_cross_dataset_evaluation.ipynb

# Download additional datasets
# TESS: https://tspace.library.utoronto.ca/
# CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D

# Load best trained model
model = tf.keras.models.load_model('results/best_cnn_lstm_model.h5')

# Function to prepare other datasets
def evaluate_on_dataset(model, dataset_name, dataset_path):
    """Evaluate model on different dataset"""
    
    X_new = []
    y_new = []
    
    # Load and preprocess dataset
    # (implementation depends on dataset format)
    
    X_new = np.array(X_new)
    X_new = X_new[:, :, :, np.newaxis]
    
    accuracy = model.evaluate(X_new, y_new)[1]
    return accuracy

# Evaluate on TESS, CREMA-D
tess_acc = evaluate_on_dataset(model, 'TESS', 'data/tess/')
cremad_acc = evaluate_on_dataset(model, 'CREMA-D', 'data/crema_d/')

print(f"RAVDESS (trained): {test_accuracy:.4f}")
print(f"TESS (zero-shot): {tess_acc:.4f}")
print(f"CREMA-D (zero-shot): {cremad_acc:.4f}")

# Report cross-dataset results
cross_dataset_results = pd.DataFrame({
    'Dataset': ['RAVDESS (trained)', 'TESS (zero-shot)', 'CREMA-D (zero-shot)'],
    'Accuracy': [test_accuracy, tess_acc, cremad_acc],
    'Drop': [0, test_accuracy - tess_acc, test_accuracy - cremad_acc]
})

print("\n=== CROSS-DATASET EVALUATION ===")
print(cross_dataset_results.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
datasets = cross_dataset_results['Dataset']
accuracies = cross_dataset_results['Accuracy']
colors = ['#2ca02c', '#ff7f0e', '#d62728']
ax.bar(datasets, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy')
ax.set_title('Cross-Dataset Generalization')
ax.set_ylim([0.7, 1.0])
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('results/cross_dataset_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Task 7.2: Hindi Extension (4 hours)
```python
# Notebook: 07_hindi_emotion_recognition.ipynb

# Option 1: Use IITKGP-SEHSC dataset
# Download from: https://ieeexplore.ieee.org/document/5738540/
# Or search: "IITKGP-SEHSC dataset" on GitHub

# Option 2: Create custom Hindi dataset
def record_hindi_emotions():
    """
    Record your own Hindi emotional speech
    
    Instructions:
    1. Use any audio recorder (Audacity, phone)
    2. Record 5 examples per emotion
    3. Emotions: Angry, Happy, Sad, Fear, Neutral
    4. Save as: hindi_[emotion]_[number].wav
    5. Sample rate: 16 kHz
    """
    pass

# Load and evaluate Hindi data
X_hindi = []
y_hindi = []

hindi_emotions = ['angry', 'happy', 'sad', 'fear', 'neutral']
hindi_emotion_to_id = {e: i for i, e in enumerate(hindi_emotions)}

for emotion in hindi_emotions:
    audio_files = glob(f'data/hindi/{emotion}/*.wav')
    for audio_file in audio_files:
        y, sr = librosa.load(audio_file, sr=16000)
        mel_spec = extract_mel_spectrogram(y, sr)
        # Pad/resize to match training size
        mel_spec_padded = pad_to_size(mel_spec, (128, 130))
        X_hindi.append(mel_spec_padded)
        y_hindi.append(hindi_emotion_to_id[emotion])

X_hindi = np.array(X_hindi)[:, :, :, np.newaxis]
y_hindi = np.array(y_hindi)

print(f"Hindi dataset size: {X_hindi.shape}")

# Evaluate pre-trained model on Hindi
hindi_accuracy = model.evaluate(X_hindi, y_hindi)[1]
print(f"English model on Hindi: {hindi_accuracy:.4f}")

# Fine-tune model for Hindi
# Freeze early layers
for layer in model.layers[:-4]:
    layer.trainable = False

# Train on Hindi data
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_hindi = model.fit(
    X_hindi[:int(0.8*len(X_hindi))], 
    y_hindi[:int(0.8*len(y_hindi))],
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

hindi_finetuned_accuracy = model.evaluate(
    X_hindi[int(0.8*len(X_hindi)):], 
    y_hindi[int(0.8*len(y_hindi)):]
)[1]

print(f"English model fine-tuned on Hindi: {hindi_finetuned_accuracy:.4f}")

# Comparison
hindi_results = pd.DataFrame({
    'Scenario': ['English model (zero-shot)', 'English model (fine-tuned 50 epochs)'],
    'Accuracy': [hindi_accuracy, hindi_finetuned_accuracy]
})

print("\n=== HINDI EMOTION RECOGNITION ===")
print(hindi_results.to_string(index=False))
```

### Deliverables for Weeks 7-8
- [ ] TESS dataset evaluated (zero-shot)
- [ ] CREMA-D dataset evaluated (zero-shot)
- [ ] Cross-dataset report (accuracy drop analysis)
- [ ] IITKGP-SEHSC Hindi data loaded
- [ ] Hindi emotion recognition model trained/fine-tuned
- [ ] Transfer learning analysis (English→Hindi)
- [ ] GitHub commit: "Week 8: Cross-dataset & Hindi extension"

---

## WEEKS 9-10: CUSTOM TEST DATASET & FINAL PRESENTATION

### Task 9.1: Create Custom Test Dataset (4 hours)
```python
# Notebook: 08_custom_test_dataset.ipynb

"""
Recording Instructions:

1. SETUP
   - Use any audio recorder (Audacity is free)
   - Sample rate: 16 kHz
   - Format: WAV
   - Record in quiet room

2. EMOTIONS TO RECORD (6)
   - Neutral: Normal voice, emotionless
   - Happy: Excited, joyful, enthusiastic
   - Sad: Depressed, melancholic, sorrowful
   - Angry: Frustrated, aggressive, angry
   - Fear: Scared, anxious, fearful (optional)
   - Surprise: Amazed, shocked, surprised (optional)

3. SCRIPT
   Record these sentences in different emotions:
   - "I love this moment"
   - "I cannot believe this happened"
   - "How beautiful is this"
   - "Hello there"
   - "Goodbye my friend"

4. RECORDING DETAILS
   - 4-5 samples per emotion per person
   - 3-5 people = 60-150 total samples
   - Keep natural emotion, not over-acting
   
5. FILE NAMING
   speaker_emotion_take.wav
   Example: alice_happy_1.wav, bob_sad_2.wav
"""

# Create directory
import os
os.makedirs('data/custom_test', exist_ok=True)

# Load custom test set
def load_custom_test_set(path='data/custom_test'):
    X_custom = []
    y_custom = []
    filenames = []
    
    emotions_map = {
        'neutral': 0, 'happy': 1, 'sad': 2, 
        'angry': 3, 'fear': 4, 'surprise': 5
    }
    
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            try:
                # Load audio
                y, sr = librosa.load(os.path.join(path, filename), sr=16000)
                
                # Extract mel-spectrogram
                mel_spec = extract_mel_spectrogram(y, sr)
                mel_spec_padded = pad_to_size(mel_spec, (128, 130))
                
                # Get emotion from filename
                emotion = filename.split('_')[1]
                emotion_id = emotions_map[emotion]
                
                X_custom.append(mel_spec_padded)
                y_custom.append(emotion_id)
                filenames.append(filename)
                
            except Exception as e:
                print(f"Error with {filename}: {e}")
    
    return np.array(X_custom)[:, :, :, np.newaxis], np.array(y_custom), filenames

X_custom, y_custom, filenames = load_custom_test_set()
print(f"Custom test set size: {X_custom.shape}")
print(f"Samples: {filenames}")

# Evaluate all trained models on custom test set
models_to_test = {
    'SVM (Classical)': svm_model,
    'Random Forest': rf_model,
    'CNN': cnn_model,
    'CNN+BiLSTM': lstm_model,
    'Ensemble+Attention': ensemble_model
}

custom_results = []

for model_name, model_obj in models_to_test.items():
    if 'Classical' in model_name or 'Forest' in model_name:
        # Classical ML: need to extract hand-crafted features
        # (re-extract for custom data)
        custom_accuracy = 0.75  # Placeholder
    else:
        # Deep learning models
        accuracy = model_obj.evaluate(X_custom, y_custom)[1]
        custom_results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Samples': len(X_custom)
        })

custom_results_df = pd.DataFrame(custom_results)
print("\n=== CUSTOM TEST SET RESULTS ===")
print(custom_results_df.to_string(index=False))
custom_results_df.to_csv('results/custom_test_results.csv', index=False)

# Detailed analysis
emotions_list = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']

for model_name, model_obj in models_to_test.items():
    if 'Deep' in type(model_obj).__name__:
        y_pred = np.argmax(model_obj.predict(X_custom), axis=1)
        
        cm = confusion_matrix(y_custom, y_pred)
        print(f"\n{model_name} Confusion Matrix:")
        print(cm)
        
        print(f"\nPer-emotion accuracy:")
        for i, emotion in enumerate(emotions_list):
            if cm[i].sum() > 0:
                accuracy = cm[i, i] / cm[i].sum()
                print(f"  {emotion}: {accuracy:.2%}")

# Visualize custom test results
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(custom_results_df['Model'], custom_results_df['Accuracy'], 
       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
       alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy')
ax.set_title(f'Performance on Custom Test Set ({len(X_custom)} samples)')
ax.set_ylim([0.6, 1.0])
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(custom_results_df['Accuracy']):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/custom_test_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Task 9.2: Generate Final Report & Presentation (6 hours)
```python
# Create: project_report.md

# PROJECT REPORT: Speech Emotion Recognition
## [Your Name], Machine Learning Course

### Executive Summary
- Objective: Build speech emotion recognition system
- Achieved: 95-97% accuracy on RAVDESS, 80-85% on custom test set
- Novelty: Ensemble with attention + cross-lingual extension
- Impact: [application potential]

### 1. Introduction
[Problem statement, motivation, applications]

### 2. Literature Review
[Summary of recent papers and approaches]

### 3. Dataset & Methodology

#### 3.1 Datasets Used
- RAVDESS: 1440 samples, 8 emotions
- TESS: 2800 samples, 7 emotions
- Custom: 150 samples, 6 emotions
- Hindi: IITKGP-SEHSC

#### 3.2 Feature Extraction
- Hand-crafted: MFCC, ZCR, RMSE, Chroma (classical ML)
- Deep learning: Mel-spectrograms (128 bins, 130 frames)

#### 3.3 Models Developed

**Classical ML:**
- SVM, Random Forest, Logistic Regression
- Accuracy: 72-82%

**Deep Learning:**
- CNN+BiLSTM: 95%
- Ensemble+Attention: 97%

### 4. Results

#### 4.1 Single Dataset Results
| Dataset | Best Model | Accuracy |
|---------|-----------|----------|
| RAVDESS | Ensemble | 97.2% |
| TESS | Ensemble | 98.1% |
| CREMA-D | Ensemble | 93.5% |

#### 4.2 Cross-Dataset Evaluation
[Table of zero-shot performance]

#### 4.3 Custom Test Set
[Results on self-recorded emotions]

#### 4.4 Hindi Language Extension
[Transfer learning results]

### 5. Analysis
- Ensemble with attention outperforms individual models
- Grad-CAM shows model focuses on relevant spectrogram regions
- Cross-dataset generalization gap: 4-8%
- Calm vs Happy easily confused

### 6. Conclusion & Future Work
[Summary, insights, next steps]

### 7. References
[All papers and datasets cited]

---

# PRESENTATION SLIDES (10 minutes)

Slide 1: Title
- Project title
- Name, date

Slide 2: Problem Statement
- Why speech emotion recognition?
- Applications

Slide 3: Approach Overview
- Classical ML → Deep Learning → Advanced Methods
- Timeline

Slide 4: Datasets
- RAVDESS/TESS/CREMA-D comparison
- Feature extraction visual

Slide 5: Classical ML Results
- SVM vs RF vs LR
- Best accuracy: 82%

Slide 6: Deep Learning Architecture
- CNN+BiLSTM diagram
- 95% accuracy

Slide 7: Advanced Method (Ensemble+Attention)
- Multi-branch architecture
- 97% accuracy
- Grad-CAM visualizations

Slide 8: Cross-Dataset & Hindi
- Zero-shot transfer results
- Hindi model fine-tuning

Slide 9: Custom Test Set
- Record emotional speech
- Real-world accuracy: 80-85%
- Confusion matrix analysis

Slide 10: Conclusion
- Key findings
- Future directions
- Demo (optional: live emotion prediction)
```

### Task 9.3: GitHub Repository Setup (2 hours)
```bash
# Create clean repository structure

SER-Project/
├── README.md
│   ├── Project description
│   ├── Quick start
│   ├── Results summary
│   ├── Datasets used
│   └── References
│
├── notebooks/
│   ├── 01_eda_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baseline_ml.ipynb
│   ├── 04_train_deep_learning.ipynb
│   ├── 05_train_advanced_methods.ipynb
│   ├── 06_cross_dataset_evaluation.ipynb
│   ├── 07_hindi_emotion_recognition.ipynb
│   └── 08_custom_test_dataset.ipynb
│
├── src/
│   ├── feature_extraction.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data_augmentation.py
│   └── utils.py
│
├── data/
│   ├── ravdess/ (download from zenodo)
│   ├── hindi/ (custom or IITKGP-SEHSC)
│   └── custom_test/ (your recorded emotions)
│
├── results/
│   ├── confusion_matrices/
│   ├── training_history.png
│   ├── model_comparison.png
│   ├── cross_dataset_evaluation.png
│   ├── gradcam_*.png
│   └── best_models/ (saved .h5 files)
│
├── requirements.txt
├── project_report.pdf
└── presentation_slides.pptx

# Initialize Git
git init
git add .
git commit -m "Initial commit: Speech Emotion Recognition Project"
git push origin main
```

### Deliverables for Weeks 9-10
- [ ] Custom test dataset created (60-150 samples, 6 emotions)
- [ ] All models evaluated on custom test set
- [ ] Per-emotion accuracy breakdown
- [ ] Final project report (5-10 pages, PDF)
- [ ] Presentation slides (10 slides)
- [ ] GitHub repository fully documented
- [ ] Model checkpoints saved
- [ ] Results visualizations (confusion matrices, accuracy plots)
- [ ] GitHub commit: "Final submission: Speech Emotion Recognition - 97% on RAVDESS, 82% on custom test"

---

## FINAL CHECKLIST

### Code Quality
- [ ] All code commented and documented
- [ ] No hardcoded paths (use config files)
- [ ] Reproducible results (set random seeds)
- [ ] Error handling for missing files

### Results Documentation
- [ ] All accuracies reported with standard deviations
- [ ] Confusion matrices for each model
- [ ] Training curves (loss and accuracy)
- [ ] Cross-dataset evaluation results

### Project Structure
- [ ] Clean file organization
- [ ] Working requirements.txt
- [ ] README with setup instructions
- [ ] All notebooks executable without errors

### Report Quality
- [ ] Clear motivation and problem statement
- [ ] Comprehensive literature review
- [ ] Detailed methodology section
- [ ] Results with tables and figures
- [ ] Error analysis and insights
- [ ] Proper citations and references

### GitHub
- [ ] Public repository with good description
- [ ] Detailed README
- [ ] All code files included
- [ ] Results and visualizations
- [ ] Model checkpoints (if not too large)

### Presentation
- [ ] 10-minute presentation ready
- [ ] Clear, readable slides
- [ ] Live demo (emotion prediction on custom audio)
- [ ] Q&A prepared

---

## EXPECTED PROJECT TIMELINE

```
Week 1:    Setup & Data (14 hrs)  ████
Week 2:    Classical ML (15 hrs)  ████
Week 3-4:  Deep Learning (18 hrs) █████
Week 5-6:  Advanced Method (11 hrs) ███
Week 7-8:  Cross-dataset & Hindi (7 hrs) ██
Week 9-10: Custom Test & Report (12 hrs) ███

Total: ~77 hours (part-time over 10 weeks)
```

---

**This roadmap ensures your project is comprehensive, well-documented, and publication-ready!**
