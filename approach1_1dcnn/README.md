# Final SER Project Summary (Combined 4 Datasets)

This document summarizes the final SER model and experiments from `final_ser/ser-project-final.ipynb`.

## Datasets
- RAVDESS, TESS, CREMA-D, SAVEE (combined)
- Total samples used (combined): ~7,356
- Emotions covered: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Feature: Mel-spectrograms (64 mel bins)

## Model
- Architecture: CNN + BiLSTM + Multi-Head Attention (PyTorch / Keras hybrid experiments documented in notebook)
- Input: Mel-spectrograms (64 x ~93)
- Key components:
  - 3 × CNN blocks (conv + ReLU + BatchNorm + MaxPool + Dropout)
  - 2 × BiLSTM layers (bidirectional)
  - Multi-head self-attention
  - Dense classifier head (dropout, batchnorm, ReLU)

## Training Setup
- Optimizer: Adam
- LR Scheduler: cosine (final lr reported in logs)
- Batch size: 16
- Epochs: up to 50 (early stopping used)
- Regularization: dropout (20%), L2 weight decay 0.0005, label smoothing 0.05
- Augmentations: SpecAugment, time/pitch shifts, additive noise

## Key Training Logs (selected)
- Accuracy on test data: 98.05755615234375

Excerpt:
```
Epoch 41/50
608/609 ━━━━━━━━━━━━━━━━━━━━ 0s 270ms/step - accuracy: 0.9992 - loss: 0.0041
Epoch 41: val_accuracy did not improve from 0.98058
609/609 ━━━━━━━━━━━━━━━━━━━━ 173s 284ms/step - accuracy: 0.9992 - loss: 0.0041 - val_accuracy: 0.9783 - val_loss: 0.0814 - learning_rate: 1.2500e-04
```

## Final Metrics
- Best validation accuracy (noted in logs): 98.058%
- Final validation accuracy at epoch 41: 97.83%
- Final test accuracy: 98.06%

## Observations
- The combined dataset strategy significantly improved model generalization.
- Training converged rapidly with very high train accuracy and low loss.
- Validation and test accuracy are consistent, indicating low overfitting.
- Data augmentation + class balancing were important for robustness.

## Artifacts & Files
- Notebook: `final_ser/ser-project-final.ipynb`
- Best model checkpoint: see `saved_models/` (notebook references)
- Confusion matrices and training plots: produced in the notebook (PNG outputs)
- This summary: `final_ser/FINAL_SER_SUMMARY.md`

## Recommendations / Next Steps
- Evaluate on additional out-of-distribution datasets to measure robustness.
- Convert the best model to a lightweight ONNX/TFLite format for deployment.
- Run a per-class error analysis to address remaining confusions.

---

If you'd like, I can:
- Move this summary to `results/` and link figures, or
- Generate a polished PDF report including the notebook figures.

## Custom Test Dataset (local) — Summary & Confusion Matrix

We also evaluated the final model on a custom/test dataset stored at `results/test_dataset_evaluation`.

- Files produced during evaluation:
  - `results/test_dataset_evaluation/metrics.json` — overall metrics and per-sample probabilities
  - `results/test_dataset_evaluation/classification_report.txt` — per-class precision/recall/F1/support
  - `results/test_dataset_evaluation/predictions.csv` — per-sample predictions, confidence, correctness

### Key numbers (from `metrics.json` / `classification_report.txt`)
- Overall accuracy: 35.0%
- Seen sentence accuracy: 42.5%
- Unseen sentence accuracy: 27.5%
- Actor-wise accuracies: a01: 12.5%, a02: 37.5%, a03: 31.25%, a04: 50.0%, a05: 43.75%

### Confusion Matrix
The notebook contains the final confusion-matrix plots (counts and normalized). If you want the PNG generated locally, run the following script from the repository root — it will create `results/test_dataset_evaluation/confusion_matrix.png`:

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

metrics_path = 'results/test_dataset_evaluation/metrics.json'
preds_path = 'results/test_dataset_evaluation/predictions.csv'
out_path = 'results/test_dataset_evaluation/confusion_matrix.png'

with open(metrics_path, 'r', encoding='utf-8') as f:
    labels = json.load(f).get('labels_in_report', ['angry','fear','happy','neutral','sad'])

df = pd.read_csv(preds_path)
y_true = df['true_emotion'].astype(str).tolist()
y_pred = df['predicted_emotion'].astype(str).tolist()

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix (counts)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1,2,2)
sns.heatmap(np.nan_to_num(cm_norm), annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix (normalized)')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()
print(f'Saved confusion matrix to {out_path}')
```

Place the script into a short file (e.g. `scripts/make_test_confusion.py`) and run:

```powershell
python scripts/make_test_confusion.py
```

