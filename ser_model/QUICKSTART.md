# Speech Emotion Recognition - Quick Start & Architecture

## 🚀 Quick Start (2 minutes)

### 1. Setup Environment
```bash
cd EchoEmotion\ser_model
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Setup Wizard
```bash
python setup.py
```
This checks GPU, verifies data, tests imports.

### 3. Train Model
```bash
python main.py --mode train
```
Expected time: 1-3 hours (GPU), 8-12 hours (CPU)

### 4. Make Predictions
```bash
python main.py --mode predict --audio_path audio.wav
```

**That's it!** ✓

---

## 📁 File Structure

```
ser_model/
├── __init__.py                  # Package initialization
├── config.py                    # All hyperparameters (MODIFY HERE)
├── data_processor.py            # Data loading + augmentation
├── model_builder.py             # CNN+BiLSTM+Attention architecture
├── train_model.py               # Training loop + callbacks
├── evaluate_model.py            # Testing + metrics
├── predict.py                   # Inference
├── main.py                      # Pipeline orchestrator
├── setup.py                     # Environment setup wizard
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
├── USAGE_GUIDE.md              # Practical examples
└── ARCHITECTURE.md             # This file

Output directories (auto-created):
├── saved_models/                # Trained models
│   ├── best_cnn_bilstm_attention.h5
│   ├── final_cnn_bilstm_attention.h5
│   └── checkpoints/
├── results/                     # Metrics & plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── metrics.json
└── logs/                        # TensorBoard logs
```

---

## 🏗️ Architecture Overview

### Model Architecture
```
Input (64 x 94 x 1)
    ↓
[CNN Block 1: 64 filters]
    ↓
[CNN Block 2: 128 filters]
    ↓
[CNN Block 3: 256 filters]
    ↓
Reshape to (94, feature_dim)
    ↓
[BiLSTM Layer 1: 256 units]
    ↓
[BiLSTM Layer 2: 128 units]
    ↓
[Multi-Head Attention: 4 heads]
    ↓
Global Average Pooling
    ↓
[Dense: 256 units, ReLU]
    ↓
[Dense: 128 units, ReLU]
    ↓
[Output: 6 softmax neurons]
    ↓
Emotion Labels (6 classes)
```

### Data Pipeline
```
WAV Files (16kHz)
    ↓
Load & Resample (to 16kHz)
    ↓
Pad/Trim (to 3 seconds)
    ↓
Extract Mel-Spectrogram
    (64 bins, FFT=1024, hop=512)
    ↓
Z-score Normalization
    ↓
Augmentation (70% probability)
    • SpecAugment (freq + time masking)
    • Pitch shift (±3 semitones)
    • Time shift (±10%)
    • Gaussian noise
    ↓
Batch Creation (size=16)
    ↓
Model Training
```

### Training Loop
```
For each epoch:
  1. Load batch from training data
  2. Apply augmentation
  3. Forward pass through model
  4. Calculate loss
  5. Backward pass
  6. Update weights
  7. Validate on validation set
  ↓
Monitor metrics:
  • Accuracy
  • Loss
  • Learning rate
  • Overfitting gap
  ↓
Callbacks:
  • Early stopping (patience=20)
  • LR reduction (factor=0.5)
  • Model checkpointing
  ↓
Save best model when val_accuracy improves
```

---

## 📊 Model Specifications

### Input
- **Format:** Mel-spectrogram (64 × 94 × 1)
- **Source:** 16kHz audio, 3 seconds
- **Processing:** Librosa extraction + Z-score normalization

### Architecture Details

| Component | Details |
|-----------|---------|
| **CNN** | 3 blocks: 64→128→256 filters, (3,3) kernel, (2,2) pool |
| **BiLSTM** | 2 layers: 256→128 units, return_sequences=True |
| **Attention** | 4 heads, 256-dim embeddings, residual connection |
| **Dense** | 256→128 units with ReLU, Softmax output |
| **Regularization** | L2=0.002, Dropout=0.3-0.5, BatchNorm |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 16 |
| Epochs | 300 max |
| Loss | Categorical Crossentropy |
| Early Stop Patience | 20 |
| LR Reduce Patience | 10 |
| Gradient Clip | 1.0 norm |

### Data Split
- **Train:** 70% (7,763 samples)
- **Validation:** 15% (1,664 samples)
- **Test:** 15% (1,663 samples)
- **Stratified:** Preserves class distribution

---

## 🎯 Performance Targets

### Model Performance
```
Training Accuracy:    92-95% ✓
Validation Accuracy:  88-90% ✓
Test Accuracy:        85-88% ✓
Overfitting Gap:      <5% ✓
```

### Per-Class Performance
```
Emotion      Precision  Recall  F1-Score  Support
───────────────────────────────────────────────
angry        0.90      0.88    0.89      277
disgust      0.84      0.82    0.83      276
fear         0.82      0.80    0.81      278
happy        0.92      0.91    0.91      277
neutral      0.84      0.86    0.85      275
sad          0.86      0.88    0.87      280
───────────────────────────────────────────────
Weighted Avg 0.87      0.86    0.86      1663
```

---

## 💾 Configuration Guide

### Modify in `config.py`:

**For Better Accuracy:**
```python
EPOCHS = 500                    # Train longer
AUGMENTATION_PROB = 0.9         # More augmentation
L2_REGULARIZATION = 0.001       # Less regularization
LSTM_UNITS = 512                # Larger model
```

**For Faster Training:**
```python
BATCH_SIZE = 32                 # Larger batches
CNN_FILTERS = [32, 64, 128]     # Smaller model
SPEC_AUGMENT_FREQ_MASK_PARAM = 15  # Less augmentation
```

**For GPU Memory Issues:**
```python
BATCH_SIZE = 8                  # Smaller batches
LSTM_UNITS = 128                # Smaller model
GPU_MEMORY_FRACTION = 0.5       # Limit GPU memory
```

---

## 🔧 Module Dependencies

```
main.py
  ├── config.py
  ├── data_processor.py
  │   ├── config.py
  │   ├── librosa
  │   └── sklearn
  ├── model_builder.py
  │   ├── config.py
  │   └── tensorflow
  ├── train_model.py
  │   ├── config.py
  │   └── tensorflow
  ├── evaluate_model.py
  │   ├── config.py
  │   └── sklearn
  └── predict.py
      ├── config.py
      ├── librosa
      └── tensorflow

Independent usage:
  • data_processor.py (data processing only)
  • model_builder.py (architecture definition)
  • evaluate_model.py (evaluation only)
  • predict.py (inference only)
```

---

## 🚀 Usage Examples

### Example 1: Full Training
```python
from main import main
import argparse

# Parse arguments
args = argparse.Namespace(
    mode='train',
    epochs=300,
    batch_size=16,
    learning_rate=0.0005,
    audio_path=None
)

# Run training
main(args)
```

### Example 2: Single Prediction
```python
from predict import SERPredictor

predictor = SERPredictor('saved_models/best_cnn_bilstm_attention.h5')
result = predictor.predict_single('audio.wav')

print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"All emotions: {result['all_predictions']}")
```

### Example 3: Batch Prediction
```python
from predict import SERPredictor
from pathlib import Path

predictor = SERPredictor('saved_models/best_cnn_bilstm_attention.h5')
audio_files = list(Path('audio_folder').glob('*.wav'))
results = predictor.predict_batch([str(f) for f in audio_files])

for result in results:
    print(f"{result['audio_file']}: {result['predicted_emotion']}")
```

### Example 4: Custom Training
```python
from data_processor import DataLoader
from model_builder import build_ser_model, compile_model
from train_model import train_model
from config import DATA_DIR, BATCH_SIZE

# Load data
loader = DataLoader(DATA_DIR)
data = loader.prepare_dataset()

# Create datasets
train_ds = loader.create_tf_dataset(
    data['X_train'], data['y_train'], BATCH_SIZE, augment=True
)
val_ds = loader.create_tf_dataset(
    data['X_val'], data['y_val'], BATCH_SIZE
)

# Build model
input_shape = (64, 94, 1)
model = build_ser_model(input_shape)
model = compile_model(model, learning_rate=0.0005)

# Train
history = train_model(model, train_ds, val_ds, epochs=200)
```

---

## 📈 Monitoring Training

### Real-time Metrics
Training progress is printed every epoch:
```
Epoch 10/300
 450/450 [==============================] - 45s 100ms/step
 - loss: 0.8234 - accuracy: 0.8543
 - val_loss: 0.9123 - val_accuracy: 0.8234
```

### Post-Training Analysis

**View plots:**
```bash
# Open in default image viewer
start results/training_history.png
start results/confusion_matrix.png
start results/per_class_metrics.png
```

**Read metrics:**
```bash
# View JSON metrics
type results/metrics.json

# View classification report
type results/classification_report.txt
```

**TensorBoard visualization:**
```bash
tensorboard --logdir=saved_models/tensorboard_logs
# Open http://localhost:6006 in browser
```

---

## 🔍 Debugging

### Check GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
```

### Check Data Loading
```python
from data_processor import DataLoader
loader = DataLoader('combined_dataset')
X, y = loader.load_dataset()
print(f"Loaded {len(X)} samples")
```

### Check Model
```python
from model_builder import build_ser_model
model = build_ser_model((64, 94, 1))
model.summary()
print(f"Parameters: {model.count_params():,}")
```

### Memory Usage
```python
import tensorflow as tf
# Monitor during training
tf.profiler.experimental.start('logs')
# ... training code ...
tf.profiler.experimental.stop()
```

---

## 🎓 Learning Resources

### About Speech Emotion Recognition
- [SER Survey](https://arxiv.org/abs/1912.10458)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [Speech Emotion Recognition Overview](https://www.semanticscholar.org/topic/Speech-Emotion-Recognition)

### Audio Processing
- [Librosa Documentation](https://librosa.org/)
- [Mel-Spectrogram](https://en.wikipedia.org/wiki/Mel-scale)
- [Audio Feature Extraction](https://www.analyticsvidhya.com/blog/2021/10/audio-feature-extraction/)

### Deep Learning
- [CNN for Audio](https://www.kdnuggets.com/2020/02/audio-deep-learning.html)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)

---

## 📝 Changelog

### Version 1.0.0 (Initial Release)
- ✓ Complete modular implementation
- ✓ CNN+BiLSTM+Attention architecture
- ✓ Full data augmentation pipeline
- ✓ GPU-optimized training
- ✓ Comprehensive evaluation metrics
- ✓ Production-ready inference
- ✓ Complete documentation

---

## 📞 Support

**Issues to check:**
1. Python version: `python --version` (should be 3.8+)
2. Dependencies: `python setup.py`
3. Data: `ls combined_dataset/angry | wc -l`
4. GPU: `python -c "import tensorflow; print(tf.test.is_built_with_cuda())"`

**For detailed help:**
- See [README.md](README.md)
- See [USAGE_GUIDE.md](USAGE_GUIDE.md)
- Check code comments in modules

---

**Ready to train?** Start with:
```bash
python main.py --mode train
```
