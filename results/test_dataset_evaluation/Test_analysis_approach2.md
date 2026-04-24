# Test Dataset Evaluation - EchoEmotion SER

This README summarizes the evaluation of our best-trained Speech Emotion Recognition (SER) model on the custom test dataset. It covers the test dataset structure, evaluation metrics, methodology, and a brief overview of the model architecture and training regime.

---

## Test Dataset Overview
- **Location:** `EchoEmotion/test_dataset/`
- **Actors:** 5 (a01, a02, a03, a04, a05)
- **Samples:** 80 audio files
- **Emotions Covered:** angry, fear, happy, neutral, sad
- **Sentence Types:**
  - `seen`: Sentences present in training
  - `unseen`: Sentences not present in training

## Evaluation Methodology
- **Model Used:** `saved_models/best_cnn_bilstm_attention.pt` (best checkpoint from training)
- **Inference:** Each audio file is processed and classified into one of the 6 emotion classes.
- **Metrics Computed:**
  - Overall accuracy
  - Actor-wise accuracy
  - Seen vs Unseen sentence accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix
  - Per-sample predictions with confidence scores

## Key Results

### Overall Metrics
- **Overall Accuracy:** 35.0%
- **Seen Sentence Accuracy:** 42.5%
- **Unseen Sentence Accuracy:** 27.5%

### Actor-wise Accuracy
| Actor | Accuracy |
|-------|----------|
| a01   | 12.5%    |
| a02   | 37.5%    |
| a03   | 31.3%    |
| a04   | 50.0%    |
| a05   | 43.8%    |

### Per-Class Metrics
```
              precision    recall  f1-score   support

       angry       1.00      0.10      0.18        20
        fear       0.00      0.00      0.00         0
       happy       0.46      0.65      0.54        20
     neutral       0.38      0.50      0.43        20
         sad       0.60      0.15      0.24        20

    accuracy                           0.35        80
   macro avg       0.49      0.28      0.28        80
weighted avg       0.61      0.35      0.35        80
```

### Confusion Matrix
- See `confusion_matrix.png` for details (if available).
- Most confusion between "angry", "happy", and "neutral".

### Seen vs Unseen Sentences
- Model performs better on sentences seen during training (42.5%) than on unseen (27.5%), indicating some generalization gap.

## Model & Training Summary
- **Architecture:** CNN + BiLSTM + Multi-Head Attention (PyTorch)
- **Input:** Mel-spectrogram features
- **Key Layers:**
  - 3x CNN blocks (with BatchNorm, Dropout)
  - 2x BiLSTM layers
  - Multi-head self-attention
  - Fully connected classifier
- **Training Regime:**
  - Combined RAVDESS, TESS, CREMA-D datasets
  - Data augmentation (SpecAugment, noise, pitch/time shift)
  - Regularization: Dropout, L2, label smoothing, class weights
  - Optimizer: Adam, cosine LR scheduler, early stopping
  - Best run: 73.98% test accuracy (on validation/test split)

## Notes & Observations
- The model achieves moderate accuracy on the custom test set, with best performance on actor a04.
- Generalization to unseen sentences is limited; further improvements may require more diverse data or advanced augmentation.
- Per-class results show high precision for "angry" but low recall, and best F1 for "happy" and "neutral".

## Files in this Folder
- `metrics.json`: All computed metrics and per-sample results
- `classification_report.txt`: Detailed per-class metrics
- `predictions.csv`: All predictions with confidence and correctness

---

For more details, see the main [TRAINING_ANALYSIS_REPORT.md](../../ser_model/TRAINING_ANALYSIS_REPORT.md).
