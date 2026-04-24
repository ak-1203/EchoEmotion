# EchoEmotion

**Speech Emotion Recognition using Deep Learning — ECL443 Course Project**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange) ![License](https://img.shields.io/badge/License-MIT-green)

**Authors:** Siddhant Jinturkar (BT23ECE066), Akash Tiwari (BT23ECE076)

---

## Table of Contents
1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Project Structure](#project-structure)
4. [Approaches](#approaches)
5. [Training Results](#training-results)
6. [Custom Dataset Evaluation](#custom-dataset-evaluation)
7. [Setup & Installation](#setup--installation)
8. [How to Use](#how-to-use)
9. [Key Findings](#key-findings)
10. [Future Work](#future-work)
11. [License](#license)
12. [Citation / Acknowledgements](#citation--acknowledgements)

---

## Overview

Speech Emotion Recognition (SER) is a challenging task aimed at classifying emotions from raw audio. This project explores two parallel approaches:

1. **Approach 1:** A 1D CNN model trained on four datasets (RAVDESS, TESS, CREMA-D, SAVEE) to classify 8 emotions.
2. **Approach 2:** A CNN+BiLSTM+Attention model trained on three datasets (RAVDESS, TESS, CREMA-D) to classify 6 emotions.

The key challenge addressed is the domain shift from studio-quality datasets to real-world audio.

---

## Datasets

| Dataset      | Emotions | Files   | Used In                  |
|--------------|----------|---------|--------------------------|
| RAVDESS      | 8        | 1440    | Phase 1, Approach 2      |
| TESS         | 7        | 2800    | Phase 1, Approach 2      |
| CREMA-D      | 6        | ~7000   | Approach 2               |
| SAVEE        | 8        | ~480    | Approach 1               |
| Combined-3   | 6        | ~11090  | Approach 2               |
| Combined-4   | 8        | ~12000+ | Approach 1               |

> **Note:** Datasets are not included in this repository. Refer to `data/README.md` for download links and instructions.

---

## Project Structure

```
EchoEmotion/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── audio_cleaning_conversion/
│
├── phase1_baselines/
│   ├── ravdess notebooks/
│   └── tess_training/
│
├── approach1_1dcnn/
│   └── README.md
│
├── approach2_cnn_bilstm/
│   └── README.md
│
├── evaluation/
│   ├── test_dataset/
│   ├── test_sample_fixed/
│   └── test_model.py
│
├── results/
│   ├── approach1_1dcnn/
│   ├── approach2_cnn-bilstm/
│   ├── phase1_ravdess/
│   ├── phase1_tess/
│   └── custom_test_evaluation/
│
├── saved_models/
│   
│
└── src/
```

---

## Approaches

### Phase 1 — Baselines (RAVDESS / TESS alone)
- **Features:** ZCR, RMSE, 40 MFCCs, augmentation ×3
- **Models:** SVM (~62%), Random Forest (~58%), CNN Mel-spec (~65%), CNN+LSTM (~70%), CNN+BiLSTM+Attention (76.2%)
- **Location:** `phase1_baselines/`

### Approach 1 — Multi-Dataset 1D CNN (Keras / TensorFlow)
- **Datasets:** RAVDESS + TESS + CREMA-D + SAVEE (~12,000 files, 8 emotions)
- **Features:** ZCR + RMSE + 40 MFCCs → 2376-dim vector
- **Architecture:** Conv1D ×3 → Dense(256) → Dense(128) → Dense(8, Softmax) | ~7.19M params
- **Training:** Adam, batch 64, early stopping
- **Results:** Test 98.05%, Val 98.06%, Macro F1 ~98%
- **Location:** `approach1_1dcnn/`
- **Results:** `results/approach1_1dcnn/`

### Approach 2 — CNN + BiLSTM + Multi-Head Attention (PyTorch)
- **Datasets:** RAVDESS + TESS + CREMA-D (~11,090 files, 6 emotions)
- **Features:** Mel-spectrogram (64 mel bins × 93 time steps)
- **Architecture:** CNN×3 → BiLSTM(256)→BiLSTM(128) → Attention(4 heads) → FC → Dense(6) | ~2.5M params
- **Training:** Adam + Cosine LR, batch 16, label smoothing 0.05, class weights, early stopping
- **Best Run:** Test 73.98%, Macro F1 74.03%
- **Location:** `approach2_cnn_bilstm/`
- **Results:** `results/approach2_cnn_bilstm/`

---

## Training Results

### Training Curves

#### Approach 1 — 1D CNN
![Approach 1 Training](results/approach1_1dcnn/training_curve.png)

#### Approach 2 — CNN + BiLSTM + Attention
![Approach 2 Training](results/approach2_cnn-bilstm/training_history.png)

### Confusion Matrices

#### Approach 1 — Multi-Dataset CNN
![Approach 1 Confusion Matrix](results/approach1_1dcnn/confusion_matrix.png)  

- **Best Test Accuracy:** 98.05%
- **Peak Validation Accuracy:** 98.06%
- **Macro F1:** ~98%

#### Approach 2 — CNN + BiLSTM + Attention
![Approach 2 Confusion Matrix](results/approach2_cnn-bilstm/confusion_matrix.png)

- Approach 1 shows near-perfect classification across all classes, indicating strong fit on combined datasets.
- Approach 2 exhibits confusion between similar emotions (e.g., neutral vs sad), reflecting real-world ambiguity and better generalization behavior. 

| Run | Epochs | Train Acc | Val Acc | Best Val | Test Acc | F1 Macro | Overfit Gap |
|-----|--------|-----------|---------|----------|----------|----------|-------------|
| R7  | 70     | 74.5%     | 76.2%   | 78.5%    | 73.98%   | 74.03%   | 2.3%        | 

### Key Observation from Visuals

Despite achieving ~98% accuracy, Approach 1 shows overly confident predictions across all classes, suggesting potential overfitting to dataset-specific patterns.

In contrast, Approach 2 demonstrates more realistic confusion patterns, which aligns with its superior performance on real-world custom data. 

---

## Custom Dataset Evaluation

- **Dataset:** 80 WAV files, 5 actors (a01–a05), 4 emotions (angry, happy, sad, neutral)
- **Domain Shift:** Studio-quality training vs real-world audio

| Method                          | Accuracy | Note                          |
|---------------------------------|----------|-------------------------------|
| Raw Inference — Approach 1 CNN | 22.5%    | Severe domain mismatch        |
| Label Mapping (7→4 class)      | 30.0%    | +7.5% gain                    |
| Probability Merging            | 28.75%   | Marginal gain                 |
| Frozen Head Fine-Tuning        | 27.0%    | Limited without unfreezing    |
| Approach 2 CNN+BiLSTM          | 35.0%    | Best on custom set            |

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip
- GPU recommended (CUDA 11+)

## Environment Setup

Recommended:
- Python 3.9 / 3.10
- NVIDIA GPU (RTX 3050 tested)

### Install PyTorch (GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

### Installation
```bash
git clone https://github.com/ak-1203/EchoEmotion
cd EchoEmotion
pip install -r requirements.txt
```

---

## How to Use

- **Datasets:** Download RAVDESS, TESS, CREMA-D, SAVEE independently. Place them in `data/`.
- **Pretrained Checkpoints:** Use `saved_models/` for evaluation.
- **Training:**
  - Approach 1: Run `approach1_1dcnn/` notebook.
  - Approach 2: Run `approach2_cnn_bilstm/` notebook.
- **Evaluation:** Run `python evaluation/test_model.py`.

---

## Key Findings

1. **Benchmark ≠ Real-World:** 73–76% on standard splits → 22–35% on custom set confirms domain shift problem.
2. **Architecture Matters:** CNN+BiLSTM+Attention (2.5M params) outperformed plain CNN+Dense (7.19M params).
3. **Transfer Learning is Key:** Label mapping +7.5%; partial-unfreeze fine-tuning is recommended.

---

## Future Work

- Expand custom dataset (more speakers, natural/diverse emotions).
- Speaker-independent cross-validation.
- Explore Transformer / wav2vec2.0 architectures.
- Aggressive SpecAugment + waveform augmentation.
- Automated hyperparameter search around R7 regime.

---

## License

This project is licensed under the MIT License.

---

## Citation / Acknowledgements

- RAVDESS (Livingstone & Russo, 2018)
- TESS, CREMA-D, SAVEE datasets
- ECL443, Machine Learning with Python, 2025–26
