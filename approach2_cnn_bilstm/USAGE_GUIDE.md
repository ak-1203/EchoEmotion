# Speech Emotion Recognition - Usage Guide (PyTorch)

Quick reference for installation, training, evaluation, and prediction.

## 1. Installation

```bash
cd c:\Users\Siddhant Jinturkar\EchoEmotion\ser_model
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional checks:

```bash
python config.py
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 2. First Run Checks

Verify dataset structure and count:

```bash
python -c "
from config import DATA_DIR
for emotion in ['angry','disgust','fear','happy','neutral','sad']:
    print(emotion, len(list((DATA_DIR/emotion).glob('*.wav'))))
"
```

Test preprocessing:

```bash
python data_processor.py
```

## 3. Training (Main Entry Point)

Use `main.py` to start training:

```bash
python main.py --mode train
```

Recommended first run uses default hyperparameters from `config.py`.

Custom examples:

```bash
python main.py --mode train --epochs 100
python main.py --mode train --batch_size 32
python main.py --mode train --learning_rate 0.001
python main.py --mode train --epochs 150 --batch_size 32 --learning_rate 0.0008
```

### Expected outputs

```text
saved_models/Approach2
|-- best_cnn_bilstm_attention.pt
|-- final_cnn_bilstm_attention.pt
|-- training_metrics.json
`-- checkpoints/

results/Approach2_cnn-bilstm
|-- training_history.png
|-- confusion_matrix.png
|-- per_class_metrics.png 
|-- metrics.json
`-- classification_report.txt
```

## 4. Evaluation

```bash
python main.py --mode evaluate
```

This loads the best saved checkpoint (or final if best is missing) and evaluates on test split.

## 5. Prediction

Single file:

```bash
python main.py --mode predict --audio_path "C:\path\to\audio.wav"
```

Multiple files:

```bash
python main.py --mode predict --audio_path "C:\audio1.wav" "C:\audio2.wav"
```

Programmatic usage:

```python
from predict import SERPredictor

predictor = SERPredictor('saved_models/best_cnn_bilstm_attention.pt')
result = predictor.predict_single('audio.wav')
print(result['predicted_emotion'], result['confidence'])
```

## 6. Advanced Programmatic Training

```python
import torch
from config import DATA_DIR, BATCH_SIZE, LEARNING_RATE
from data_processor import DataLoader
from model_builder import build_ser_model
from train_model import train_model, evaluate_training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(DATA_DIR)
data = loader.prepare_dataset()

train_ds = loader.create_torch_dataset(data['X_train'], data['y_train'], BATCH_SIZE, augment=True, shuffle=True)
val_ds = loader.create_torch_dataset(data['X_val'], data['y_val'], BATCH_SIZE, augment=False, shuffle=False)

input_shape = (1, data['X_train'].shape[1], data['X_train'].shape[2])
model = build_ser_model(input_shape)

history = train_model(
    model,
    train_ds,
    val_ds,
    epochs=200,
    learning_rate=LEARNING_RATE,
    device=device,
    scaler=data['scaler'],
    input_shape=input_shape,
)

summary = evaluate_training(history)
print(summary['best_val_accuracy'])
```

## 7. Best Results Checklist

1. Use GPU if available.
2. Start with default config and complete one full run.
3. Confirm train accuracy > 90% and test > 80/85%.
4. Check overfitting gap from summary.
5. Tune `batch_size`, `learning_rate`, and `epochs` gradually.

## 8. Troubleshooting

### CUDA OOM

- Reduce `BATCH_SIZE` in `config.py`.
- Lower `LSTM_UNITS` or `CNN_FILTERS`.

### Low accuracy

- Increase `EPOCHS`.
- Tune `LEARNING_RATE`.
- Increase `AUGMENTATION_PROB`.
- Verify class distribution in `combined_dataset`.

### Model load mismatch

- Use `.pt` checkpoints from `saved_models/`.
- Keep code and checkpoint version aligned.

For full details, see `README.md`.
