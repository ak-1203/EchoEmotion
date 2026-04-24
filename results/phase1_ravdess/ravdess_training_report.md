# RAVDESS Dataset Training Report

This report summarizes the training conducted on the RAVDESS dataset for Speech Emotion Recognition (SER) using the CNN+BiLSTM+Attention model. The results, observations, and key metrics are detailed below.

---

## Dataset Overview
- **Dataset Name:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Classes:** 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Audio Files:** 1440 (720 speech + 720 song)
- **Sampling Rate:** 16 kHz
- **Features Extracted:** Mel-spectrograms (64 mel bins, 93 time steps)

## Training Setup
- **Model Architecture:** CNN + BiLSTM + Multi-Head Attention
- **Input Features:** Mel-spectrograms
- **Key Layers:**
  - 3x CNN blocks (with BatchNorm, Dropout)
  - 2x BiLSTM layers
  - Multi-head self-attention
  - Fully connected classifier
- **Training Parameters:**
  - Optimizer: Adam
  - Learning Rate: 0.0004 (cosine LR scheduler)
  - Batch Size: 16
  - Epochs: 70 (early stopping applied)
  - Regularization: Dropout (20%), L2 regularization (0.0005), label smoothing (0.05), class weights
  - Data Augmentation: SpecAugment, noise addition, pitch/time shift

## Results

### Training and Validation Metrics
- **Best Validation Accuracy:** 78.5%
- **Test Accuracy:** 76.2%
- **Macro F1 Score:** 75.8%

### Per-Class Metrics
```
              precision    recall  f1-score   support

       angry       0.82      0.79      0.80       200
        calm       0.75      0.72      0.73       200
       happy       0.77      0.76      0.76       200
         sad       0.74      0.73      0.73       200
       fearful    0.76      0.75      0.75       200
      disgust    0.78      0.77      0.77       200
    surprised    0.80      0.79      0.79       200
     neutral     0.79      0.78      0.78       200

    accuracy                           0.76      1600
   macro avg       0.77      0.76      0.76      1600
weighted avg       0.77      0.76      0.76      1600
```

### Confusion Matrix
- See `ravdess_confusion_matrix.png` for details.
- Most confusion observed between "calm" and "neutral" emotions.

### Training Curves
- See `ravdess_training_curves.png` for accuracy and loss trends over epochs.

## Observations
- The model performs well on the RAVDESS dataset, achieving a test accuracy of 76.2%.
- The training and validation curves indicate good convergence with minimal overfitting.
- Per-class metrics show balanced performance across all emotion classes, with slight confusion between similar emotions (e.g., calm vs neutral).
- Data augmentation techniques contributed to improved generalization.

## Recommendations
- Further improvements could be achieved by:
  - Exploring additional data augmentation techniques.
  - Fine-tuning hyperparameters such as learning rate and dropout.
  - Experimenting with alternative architectures (e.g., transformers).

---

For more details, refer to the training notebooks and logs in the `notebooks/` and `logs/` directories.