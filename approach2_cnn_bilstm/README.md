# Speech Emotion Recognition (SER) - PyTorch Implementation

A modular, production-ready Speech Emotion Recognition system using a `CNN + BiLSTM + Multi-Head Attention` architecture.

## Highlights

- PyTorch-based training and inference
- Automatic GPU usage when available
- Combined dataset support (RAVDESS + TESS + CREMA-D)
- 6 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`
- Train/Val/Test split with augmentation and reporting

## Project Structure

```text
appraoch2_cnn_bilstm/
|-- config.py
|-- data_processor.py
|-- model_builder.py
|-- train_model.py
|-- evaluate_model.py
|-- predict.py
|-- main.py
|-- requirements.txt
|-- README.md
|-- USAGE_GUIDE.md
```

Outputs (auto-created):

```text
saved_models/Approach2/
|-- best_cnn_bilstm_attention.pt
|-- final_cnn_bilstm_attention.pt
|-- training_metrics.json
|-- checkpoints/

results/
|-- training_history.png
|-- confusion_matrix.png
|-- per_class_metrics.png
|-- metrics.json
|-- classification_report.txt
```

## Installation

```bash
cd ser_model
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import librosa; print('Librosa OK')"
```

## Dataset Layout

Expected structure:

```text
combined_dataset/
|-- angry/
|-- disgust/
|-- fear/
|-- happy/
|-- neutral/
`-- sad/
```

Each folder should contain `.wav` files.

## Quick Start

### Start Training (recommended)

```bash
cd ser_model
python main.py --mode train
```

This does:
1. Load and preprocess audio
2. Create 70/15/15 splits
3. Train model with early stopping and LR scheduler
4. Save best/final checkpoints (`.pt`)
5. Evaluate test set and save reports

### Custom Training

```bash
python main.py --mode train --epochs 200 --batch_size 32 --learning_rate 0.001
```

### Evaluate Saved Model

```bash
python main.py --mode evaluate
```

### Predict Emotion

```bash
python main.py --mode predict --audio_path C:\path\to\audio.wav
python main.py --mode predict --audio_path C:\a.wav C:\b.wav
```

## Performance Status

Current best achieved in this repository (latest run):

- Test Accuracy: `73.98%`
- Macro F1: `74.03%`
- Best Validation Accuracy: `76.98%`

Target goals remain:

- Train Accuracy: `> 90%`
- Test Accuracy: `> 80-85%`
- Overfitting gap: ideally `< 5%`

Performance depends on data quality, class balance, augmentation, and hyperparameters.

## Best-Results Tips

1. Use GPU (`torch.cuda.is_available() == True`).
2. Start with default settings first (`--mode train`).
3. If GPU memory allows, try `--batch_size 32`.
4. If underfitting, increase epochs.
5. If overfitting, reduce learning rate or increase augmentation.

## Troubleshooting

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### CUDA out of memory

- Lower `BATCH_SIZE` in `config.py` (e.g., 8)
- Lower model size (`LSTM_UNITS`, `CNN_FILTERS`)

### Slow loading

- Keep dataset on SSD
- Increase `NUM_WORKERS` in `config.py`

## Notes

- Model files are now saved as `.pt` (not `.h5`).
- This repository is now documented for PyTorch workflow.
