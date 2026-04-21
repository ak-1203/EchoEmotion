# Speech Emotion Recognition (SER) - Production-Ready Implementation

A complete, modular, and production-ready deep learning system for Speech Emotion Recognition using CNN+BiLSTM+Attention architecture.

## Features

✅ **Modular Architecture** - 7 independent, reusable modules
✅ **CNN+BiLSTM+Attention** - State-of-the-art neural network design
✅ **Data Augmentation** - SpecAugment, pitch shift, time shift, noise
✅ **GPU Optimized** - Efficient training with TensorFlow/Keras
✅ **Production Ready** - Type hints, logging, error handling
✅ **6 Emotion Classes** - angry, disgust, fear, happy, neutral, sad
✅ **~11,000 Audio Files** - Combined RAVDESS, TESS, CREMA-D datasets

## Project Structure

```
ser_model/
├── config.py              # Centralized configuration & hyperparameters
├── data_processor.py      # Data loading, preprocessing, augmentation
├── model_builder.py       # CNN+BiLSTM+Attention model architecture
├── train_model.py         # Training pipeline with callbacks
├── evaluate_model.py      # Testing & comprehensive metrics
├── predict.py             # Single/batch audio prediction
├── main.py                # Complete pipeline orchestrator
├── requirements.txt       # Python dependencies
└── README.md              # This file

Output directories (auto-created):
├── saved_models/          # Trained models & checkpoints
├── results/               # Plots, metrics, reports
└── logs/                  # Training logs & TensorBoard
```

## Installation

### 1. Create Virtual Environment

```bash
# Using Python venv
python -m venv ser_env
source ser_env/Scripts/activate  # Windows: ser_env\Scripts\activate.bat

# Or using conda
conda create -n ser_env python=3.10
conda activate ser_env
```

### 2. Install Dependencies

```bash
cd ser_model
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import librosa; print('Librosa OK')"
```

## Dataset Preparation

The implementation expects audio files organized as:

```
combined_dataset/
├── angry/
│   ├── ravdess_03-01-05-01-01-01-01.wav
│   ├── tess_ANGER.wav
│   └── ...
├── disgust/
├── fear/
├── happy/
├── neutral/
└── sad/
```

**Expected statistics:**
- ~11,090 WAV files
- 6 emotion classes (balanced)
- Sample rate: 16 kHz (auto-resampled)
- Duration: 0.5-10 seconds (auto-trimmed to 3s)

## Usage

### Quick Start - Full Training Pipeline

```bash
cd ser_model
python main.py --mode train
```

This will:
1. Load and preprocess all audio files
2. Create 70/15/15 train/val/test split
3. Train CNN+BiLSTM+Attention model (max 300 epochs)
4. Evaluate on test set with detailed metrics
5. Save best model to `saved_models/best_cnn_bilstm_attention.h5`
6. Generate plots and classification reports

### Training with Custom Hyperparameters

```bash
python main.py --mode train --epochs 200 --batch_size 32 --learning_rate 0.001
```

### Evaluate Existing Model

```bash
python main.py --mode evaluate
```

Loads the best saved model and evaluates on test set.

### Make Predictions

```bash
# Single audio file
python main.py --mode predict --audio_path /path/to/audio.wav

# Multiple files
python main.py --mode predict --audio_path /path/to/audio1.wav /path/to/audio2.wav
```

## Architecture Details

### Model Components

#### 1. **CNN Feature Extraction**
- 3 convolutional blocks
- Filters: 64 → 128 → 256
- Kernel: (3,3), Pool: (2,2)
- BatchNorm + Dropout (0.3) after each block

#### 2. **BiLSTM Temporal Processing**
- 2 bidirectional LSTM layers
- Units: 256 → 128
- Dropout: 0.4, Recurrent Dropout: 0.2
- Captures temporal dependencies in audio

#### 3. **Multi-Head Attention**
- 4 attention heads
- 256-dimensional embeddings
- Residual connections
- Learns which time steps are important

#### 4. **Dense Classification**
- Global average pooling over time
- 2 dense layers (256 → 128 units)
- ReLU activation with BatchNorm
- Final softmax (6 outputs)

#### 5. **Regularization**
- L2 regularization: 0.002
- Dropout: 0.3-0.5
- BatchNormalization
- Early stopping with patience=20

### Input Specification

- **Shape:** (64 mel bins, 94 time steps, 1 channel)
- **Duration:** 3 seconds at 16kHz
- **Mel-spectrogram:** 64 bins, FFT=1024, hop=512
- **Normalization:** Z-score (per sample)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Max Epochs | 300 |
| Early Stopping Patience | 20 |
| LR Reduce Factor | 0.5 |
| LR Reduce Patience | 10 |

## Data Augmentation

**Training data augmentation includes:**

1. **SpecAugment**
   - Frequency masking (30 bins)
   - Time masking (40 frames)
   - Applied with 70% probability

2. **Time Shift**
   - Random shift ±10% of duration
   - Preserves circular nature

3. **Pitch Shift**
   - ±3 semitones
   - Librosa implementation

4. **Gaussian Noise**
   - σ = 0.005
   - Added to spectrogram

Augmentation is **only applied to training data**, not validation/test.

## Expected Performance

| Metric | Expected | Target |
|--------|----------|--------|
| Train Accuracy | 92-95% | >90% |
| Test Accuracy | 85-88% | >85% |
| F1-Score (macro) | 84-87% | >80% |
| Overfitting Gap | <5% | <5% |

**Per-class F1 scores typically:**
- Angry: 88-92%
- Disgust: 82-86%
- Fear: 80-84%
- Happy: 90-94%
- Neutral: 82-86%
- Sad: 84-88%

## Output Files

### After Training

```
saved_models/
├── best_cnn_bilstm_attention.h5      # Best model (lowest val loss)
├── final_cnn_bilstm_attention.h5     # Final model (after 300 epochs)
├── training_metrics.json              # Training history
└── checkpoints/
    └── model_epoch_XXX.h5            # Periodic checkpoints

results/
├── training_history.png               # Loss/Accuracy curves
├── confusion_matrix.png               # Absolute & normalized
├── per_class_metrics.png              # Per-emotion F1/Accuracy
├── metrics.json                       # Detailed test metrics
└── classification_report.txt          # Sklearn classification report

logs/
└── [tensorboard logs]
```

## Configuration

All hyperparameters are centralized in `config.py`:

```python
# Audio Processing
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Model Architecture
CNN_FILTERS = [64, 128, 256]
LSTM_UNITS = 256
NUM_ATTENTION_HEADS = 4
L2_REGULARIZATION = 0.002

# Training
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 20

# Data Split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

Modify these values to experiment with different configurations.

## Module Documentation

### 1. `config.py`
- Centralized configuration
- Path management
- Hyperparameter definitions
- Emotion mappings

### 2. `data_processor.py`
- `AudioProcessor`: Load & extract mel-spectrograms
- `DataAugmentor`: SpecAugment, pitch shift, noise
- `DataLoader`: Dataset management & TF pipeline

**Key methods:**
```python
loader = DataLoader(data_dir)
data = loader.prepare_dataset()  # Returns train/val/test split
train_ds = loader.create_tf_dataset(X_train, y_train, batch_size, augment=True)
```

### 3. `model_builder.py`
- `MultiHeadAttention`: Custom attention layer
- `CNNBlock`: Reusable conv block
- `build_ser_model()`: Constructs full architecture
- `compile_model()`: Compiles with optimizer

**Usage:**
```python
model = build_ser_model(input_shape=(64, 94, 1))
model = compile_model(model, learning_rate=0.0005)
model.summary()
```

### 4. `train_model.py`
- `CustomMetricsCallback`: Track metrics
- `TrainingMonitor`: Plot generation
- `train_model()`: Full training pipeline
- `evaluate_training()`: Summary statistics

**Usage:**
```python
history = train_model(model, train_dataset, val_dataset, epochs=300)
summary = evaluate_training(history)
```

### 5. `evaluate_model.py`
- `ModelEvaluator`: Comprehensive evaluation
- `generate_evaluation_report()`: Full report generation
- Confusion matrix, per-class metrics, classification report

**Usage:**
```python
evaluator = ModelEvaluator(model)
results = evaluator.evaluate(test_dataset)
evaluator.plot_confusion_matrix(results['y_true'], results['y_pred'])
```

### 6. `predict.py`
- `SERPredictor`: Inference class
- `predict_single()`: Single audio
- `predict_batch()`: Multiple audios
- `load_and_predict()`: Convenience function

**Usage:**
```python
predictor = SERPredictor(model_path='saved_models/best_model.h5')
result = predictor.predict_single('audio.wav')
print(result['predicted_emotion'], result['confidence'])
```

### 7. `main.py`
- Complete pipeline orchestrator
- CLI argument parsing
- Three modes: train, evaluate, predict
- Environment setup (GPU, reproducibility)

## Advanced Usage

### Custom Training Loop

```python
from data_processor import DataLoader
from model_builder import build_ser_model, compile_model
from train_model import train_model

# Load data
loader = DataLoader('path/to/data')
data = loader.prepare_dataset()
train_ds = loader.create_tf_dataset(data['X_train'], data['y_train'], 16, augment=True)
val_ds = loader.create_tf_dataset(data['X_val'], data['y_val'], 16, augment=False)

# Build model
input_shape = (64, 94, 1)
model = build_ser_model(input_shape)
model = compile_model(model, learning_rate=0.0005)

# Train
history = train_model(model, train_ds, val_ds, epochs=200)
```

### Batch Prediction

```python
from predict import SERPredictor
from pathlib import Path

predictor = SERPredictor('saved_models/best_model.h5')
audio_files = list(Path('audio_folder').glob('*.wav'))
results = predictor.predict_batch([str(f) for f in audio_files])

for result in results:
    print(f"{result['audio_file']}: {result['predicted_emotion']} ({result['confidence']:.3f})")
```

### Load & Fine-tune

```python
import tensorflow as tf
from model_builder import MultiHeadAttention

# Load
model = tf.keras.models.load_model(
    'saved_models/best_model.h5',
    custom_objects={'MultiHeadAttention': MultiHeadAttention}
)

# Fine-tune (freeze CNN, unfreeze LSTM+Attention)
for layer in model.layers[:9]:  # CNN blocks
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training on new data
# model.fit(new_train_ds, validation_data=new_val_ds, epochs=50)
```

## Troubleshooting

### GPU Not Detected

```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Out of Memory (OOM)

1. Reduce `BATCH_SIZE` in config.py (e.g., 8 instead of 16)
2. Reduce `LSTM_UNITS` (e.g., 128 instead of 256)
3. Reduce `CNN_FILTERS` (e.g., [32, 64, 128])
4. Set `ALLOW_GROWTH = True` and lower `GPU_MEMORY_FRACTION`

### Audio Loading Errors

1. Ensure audio files are valid WAV format
2. Check file permissions
3. Install audioread: `pip install audioread`
4. On Windows, install FFmpeg: `choco install ffmpeg`

### Poor Model Performance

1. Check dataset balance: `python data_processor.py`
2. Increase augmentation probability
3. Reduce L2 regularization (0.001 instead of 0.002)
4. Increase learning rate (0.001 instead of 0.0005)
5. Train for more epochs (500 instead of 300)

## Performance Optimization

### Faster Training

1. Increase batch size (32 or 64)
2. Reduce model complexity (fewer filters/units)
3. Use mixed precision: `tf.keras.mixed_precision.Policy('mixed_float16')`
4. Enable XLA compilation: `@tf.function(jit_compile=True)`

### Better Accuracy

1. Use ensemble methods (train 5 models, average predictions)
2. Increase training data with more augmentation
3. Use learning rate warmup (from 1e-5 to 5e-4 over first 5 epochs)
4. Implement class weighting for imbalanced data
5. Cross-validate (5-fold CV)

## Citing This Work

```bibtex
@software{ser_model_2024,
  title={Speech Emotion Recognition: Production-Ready CNN+BiLSTM+Attention},
  author={EchoEmotion},
  year={2024}
}
```

## License

This implementation is provided as-is for research and production use.

## Support

For issues, questions, or contributions:
1. Check this README thoroughly
2. Review code comments and docstrings
3. Check TensorFlow/Librosa documentation
4. Inspect saved logs in `logs/` directory
5. Validate data with `python data_processor.py`

## References

- **SpecAugment**: Park et al., 2019 (Data Augmentation for Speech Recognition)
- **Attention is All You Need**: Vaswani et al., 2017
- **CNN+LSTM for Audio**: Lim & Choi, 2016
- **Mel-Spectrogram**: Librosa Documentation

---

**Last Updated:** 2024
**Python Version:** 3.8+
**TensorFlow Version:** 2.13+
