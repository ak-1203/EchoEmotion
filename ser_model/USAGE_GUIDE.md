# Speech Emotion Recognition - Usage Guide

Quick reference for common tasks and workflows.

## Table of Contents

1. [Installation](#installation)
2. [First Run](#first-run)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Prediction](#prediction)
6. [Advanced Usage](#advanced-usage)

---

## Installation

### Step 1: Navigate to Project

```bash
cd c:\Users\akash\Desktop\firstFolder\ml_project\ecl443\EchoEmotion\ser_model
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
python config.py
```

Expected output shows configuration summary and paths.

---

## First Run

### Verify Data

Check that combined_dataset has audio files:

```bash
python -c "
from pathlib import Path
from config import DATA_DIR
for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']:
    emotion_dir = DATA_DIR / emotion
    files = list(emotion_dir.glob('*.wav'))
    print(f'{emotion}: {len(files)} files')
"
```

Expected output:
```
angry: ~1700 files
disgust: ~1700 files
fear: ~1700 files
happy: ~1700 files
neutral: ~1700 files
sad: ~1700 files
```

### Test Data Loading

```bash
python data_processor.py
```

This will:
- Load all audio files
- Extract mel-spectrograms
- Create train/val/test splits
- Print dataset statistics

Expected:
```
Loaded 11090 spectrograms from 6 emotion classes
Train set: (7763, 64, 94)
Validation set: (1664, 64, 94)
Test set: (1663, 64, 94)
```

---

## Training

### Full Training (Recommended First Run)

```bash
python main.py --mode train
```

**What happens:**
1. Loads all ~11,000 audio files (~5-10 minutes)
2. Extracts mel-spectrograms and normalizes
3. Creates train/val/test split (70/15/15)
4. Builds CNN+BiLSTM+Attention model
5. Trains for up to 300 epochs with callbacks
6. Saves best model automatically
7. Evaluates on test set with detailed metrics
8. Generates plots and reports

**Expected time:** 1-3 hours (depends on GPU)

**Expected accuracy:** 
- Train: 92-95%
- Test: 85-88%

### Training with Custom Parameters

```bash
# Quick training (fewer epochs)
python main.py --mode train --epochs 100

# Larger batches (faster but needs more GPU memory)
python main.py --mode train --batch_size 32

# Custom learning rate
python main.py --mode train --learning_rate 0.001

# All together
python main.py --mode train --epochs 150 --batch_size 32 --learning_rate 0.0008
```

### Training Output

After training completes, check:

```
saved_models/
├── best_cnn_bilstm_attention.h5       ← Best model
├── final_cnn_bilstm_attention.h5
├── training_metrics.json
└── checkpoints/                       ← Intermediate checkpoints

results/
├── training_history.png               ← Loss/Accuracy curves
├── confusion_matrix.png               ← Predictions by emotion
├── per_class_metrics.png              ← F1-Score per emotion
├── metrics.json                       ← Detailed test metrics
└── classification_report.txt          ← Sklearn report
```

**View Training History:**
```bash
# Open saved plot
start results/training_history.png
```

---

## Evaluation

### Quick Evaluation (Using Saved Model)

```bash
python main.py --mode evaluate
```

Loads the best trained model and evaluates on test set only.

### Manual Evaluation

```python
import tensorflow as tf
import numpy as np
from data_processor import DataLoader
from config import DATA_DIR, BATCH_SIZE, BEST_MODEL_PATH
from model_builder import MultiHeadAttention
from evaluate_model import generate_evaluation_report

# Load data
loader = DataLoader(DATA_DIR)
data = loader.prepare_dataset()
test_ds = loader.create_tf_dataset(data['X_test'], data['y_test'], BATCH_SIZE)

# Load model
model = tf.keras.models.load_model(
    str(BEST_MODEL_PATH),
    custom_objects={'MultiHeadAttention': MultiHeadAttention}
)

# Evaluate
results = generate_evaluation_report(model, test_ds)
print(f"Accuracy: {results['metrics']['overall_accuracy']:.4f}")
```

---

## Prediction

### Predict Single Audio

```bash
python main.py --mode predict --audio_path "C:\path\to\audio.wav"
```

Output:
```
======================================================================
Audio File: C:\path\to\audio.wav
======================================================================
Predicted Emotion: happy
Confidence: 0.9234 (92.34%)
Confidence Threshold Met: True

All Predictions:
  happy        : ██████████████████████████████ 0.9234
  neutral      : ██                             0.0456
  angry        : ██                             0.0234
  disgust      : █                              0.0045
  fear         :                                0.0015
  sad          :                                0.0016

Top K Predictions:
  1. happy       : 0.9234
  2. neutral     : 0.0456
  3. angry       : 0.0234
======================================================================
```

### Predict Multiple Audios

```bash
python main.py --mode predict --audio_path "C:\audio1.wav" "C:\audio2.wav" "C:\audio3.wav"
```

### Programmatic Prediction

```python
from predict import SERPredictor

# Initialize predictor
predictor = SERPredictor('saved_models/best_cnn_bilstm_attention.h5')

# Predict single file
result = predictor.predict_single('audio.wav')
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.4f}")

# Get all predictions
for emotion, confidence in result['all_predictions'].items():
    print(f"  {emotion}: {confidence:.4f}")

# Predict multiple files
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = predictor.predict_batch(audio_files)
for result in results:
    print(f"{result['audio_file']}: {result['predicted_emotion']}")
```

---

## Advanced Usage

### 1. Custom Model Training with Full Control

```python
from data_processor import DataLoader
from model_builder import build_ser_model, compile_model
from train_model import train_model, evaluate_training
from config import DATA_DIR, BATCH_SIZE, LEARNING_RATE

# Load and prepare data
loader = DataLoader(DATA_DIR)
data = loader.prepare_dataset()

# Create TF datasets
train_ds = loader.create_tf_dataset(
    data['X_train'], data['y_train'], BATCH_SIZE, augment=True
)
val_ds = loader.create_tf_dataset(
    data['X_val'], data['y_val'], BATCH_SIZE
)
test_ds = loader.create_tf_dataset(
    data['X_test'], data['y_test'], BATCH_SIZE
)

# Build and compile model
input_shape = (data['X_train'].shape[1], data['X_train'].shape[2], 1)
model = build_ser_model(input_shape)
model = compile_model(model, learning_rate=LEARNING_RATE)

# Train
history = train_model(model, train_ds, val_ds, epochs=200)

# Evaluate
summary = evaluate_training(history)

print(f"Best Val Accuracy: {summary['best_val_accuracy']:.4f}")
```

### 2. Data Analysis

```python
from data_processor import DataLoader
from config import DATA_DIR
import matplotlib.pyplot as plt
import numpy as np

loader = DataLoader(DATA_DIR)
X, y = loader.load_dataset()

# Statistics
X = np.array(X)
print(f"Total samples: {len(X)}")
print(f"Shape per sample: {X.shape[1:]}")
print(f"Min mel-spec value: {X.min():.2f}")
print(f"Max mel-spec value: {X.max():.2f}")

# Class distribution
unique, counts = np.unique(y, return_counts=True)
for idx, count in zip(unique, counts):
    from config import IDX_TO_EMOTION
    emotion = IDX_TO_EMOTION[idx]
    print(f"{emotion}: {count} samples")
```

### 3. Ensemble Predictions

Train multiple models and average their predictions:

```python
from predict import SERPredictor
import numpy as np

# Load 3 trained models
models = [
    SERPredictor('saved_models/best_model_1.h5'),
    SERPredictor('saved_models/best_model_2.h5'),
    SERPredictor('saved_models/best_model_3.h5'),
]

# Predict with all models
audio_file = 'audio.wav'
predictions = [m.predict_single(audio_file) for m in models]

# Average probabilities
avg_probs = np.mean([p['all_predictions'].values() for p in predictions], axis=0)

# Get ensemble prediction
from config import EMOTIONS
ensemble_emotion = EMOTIONS[np.argmax(avg_probs)]
ensemble_confidence = np.max(avg_probs)

print(f"Ensemble: {ensemble_emotion} ({ensemble_confidence:.4f})")
```

### 4. Fine-tuning on New Data

```python
import tensorflow as tf
from model_builder import MultiHeadAttention
from data_processor import DataLoader
from config import BEST_MODEL_PATH, DATA_DIR, BATCH_SIZE

# Load pretrained model
model = tf.keras.models.load_model(
    str(BEST_MODEL_PATH),
    custom_objects={'MultiHeadAttention': MultiHeadAttention}
)

# Freeze early layers (CNN)
for layer in model.layers[:9]:
    layer.trainable = False

# Compile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune on new data
loader = DataLoader(DATA_DIR)
data = loader.prepare_dataset()

train_ds = loader.create_tf_dataset(data['X_train'], data['y_train'], BATCH_SIZE)
val_ds = loader.create_tf_dataset(data['X_val'], data['y_val'], BATCH_SIZE)

# Train for fewer epochs
model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=1)

# Save fine-tuned model
model.save('saved_models/finetuned_model.h5')
```

### 5. Real-time Audio Processing

```python
import pyaudio
import numpy as np
from predict import SERPredictor
import librosa

# Setup
predictor = SERPredictor()
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Record 3-second audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1,
                rate=SAMPLE_RATE, input=True,
                frames_per_buffer=1024)

print("Recording 3 seconds...")
frames = []
for _ in range(int(CHUNK_SIZE / 1024)):
    data = stream.read(1024)
    frames.append(np.frombuffer(data, np.float32))

audio_data = np.concatenate(frames)

# Predict
result = predictor.predict_single('temp_audio.wav')
print(f"Detected emotion: {result['predicted_emotion']}")

stream.close()
p.terminate()
```

### 6. Model Analysis

```python
import tensorflow as tf
from model_builder import MultiHeadAttention
from config import BEST_MODEL_PATH

# Load model
model = tf.keras.models.load_model(
    str(BEST_MODEL_PATH),
    custom_objects={'MultiHeadAttention': MultiHeadAttention}
)

# Print architecture
model.summary()

# Get total parameters
total_params = model.count_params()
trainable_params = sum(w.size for w in model.trainable_weights)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Layer analysis
for i, layer in enumerate(model.layers):
    print(f"{i:2d}. {layer.name:30s} {str(layer.output_shape):40s} "
          f"{layer.count_params():,}")
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# In config.py
BATCH_SIZE = 8  # Reduce from 16
GPU_MEMORY_FRACTION = 0.7  # Reduce from 0.9
```

### Issue: Slow Data Loading

**Solution:**
```bash
# Check disk speed - move data to SSD if possible
# Or increase prefetch buffer in data_processor.py:
# dataset = dataset.prefetch(1000)
```

### Issue: Poor Model Accuracy

**Solution:**
```python
# In config.py
AUGMENTATION_PROB = 0.9  # Increase augmentation
L2_REGULARIZATION = 0.001  # Reduce regularization
EPOCHS = 500  # Train longer
```

### Issue: Model won't load

**Solution:**
```python
# Make sure custom objects are imported
from model_builder import MultiHeadAttention

model = tf.keras.models.load_model(
    'best_model.h5',
    custom_objects={'MultiHeadAttention': MultiHeadAttention}
)
```

---

## Performance Tips

1. **GPU Utilization:**
   - Batch size 16 → 32 (if memory allows)
   - Mixed precision training

2. **Training Speed:**
   - Use pretrained weights if available
   - Reduce mel-spectrogram resolution (64→32 bins)
   - Use parallel data loading

3. **Better Accuracy:**
   - Ensemble multiple models
   - Increase training data
   - Augmentation with more aggressive parameters
   - Cross-validation

---

## Next Steps

1. ✅ Complete first training run
2. ✅ Check results and metrics
3. ✅ Make predictions on test audio
4. ✅ Fine-tune on specific dataset
5. ✅ Deploy in production

For detailed information, see [README.md](README.md).
