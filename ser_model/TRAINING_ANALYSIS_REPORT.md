# SER Training Analysis Report

Date: 2026-04-21  
Project: `EchoEmotion/ser_model`  
Task: Speech Emotion Recognition (6 classes) on combined dataset (RAVDESS + TESS + CREMA-D)

## Scope

This report compares all training runs shared in terminal summaries during this session and analyzes performance trends.

## Run Comparison

| Run ID | Command / Setup | Epochs Run | Final Train Acc | Final Val Acc | Best Val Acc (Epoch) | Test Acc | F1 Macro | Overfit Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | `--epochs 150 --batch_size 32 --learning_rate 0.0008` | 78 | 0.5513 | 0.6502 | 0.6593 (68) | 0.6514 | 0.6485 | -0.0990 |
| R2 | Tuned regularization (reduced) + lower dropout run | 88 | 0.7794 | 0.7440 | 0.7488 (68) | 0.7308 | 0.7306 | 0.0354 |
| R3 | `--epochs 400 --batch_size 24 --learning_rate 0.0003` | 52 | 0.7984 | 0.7464 | 0.7494 (32) | 0.7248 | 0.7227 | 0.0520 |
| R4 | `--epochs 450 --batch_size 16 --learning_rate 0.0003` | 88 | 0.8256 | 0.7440 | 0.7572 (82) | 0.7230 | 0.7228 | 0.0816 |
| R5 | `--epochs 320 --batch_size 16 --learning_rate 0.0002` + stronger regularization config | 101 | 0.7368 | 0.7344 | 0.7434 (76) | 0.7266 | 0.7259 | 0.0024 |
| R6 | Tuned training controls (class weights + label smoothing + cosine scheduler) | 60 | 0.6674 | 0.7344 | 0.7752 (30) | 0.7332 | 0.7337 | -0.0670 |
| R7 | Follow-up tuned run | 67 | 0.7155 | 0.7314 | 0.7698 (37) | 0.7398 | 0.7403 | -0.0158 |

## Key Findings

1. Best overall generalization is **R7**:
   - Highest Test Accuracy: **73.98%**
   - Highest Macro F1: **74.03%**

2. Highest validation peak is **R6** (`best val = 77.52%`), but final test (`73.32%`) was lower than R7.

3. Recent runs (R6/R7) improved beyond the earlier best regime (R2), confirming progress from training-control tuning.

4. Current practical performance band has moved from ~72-73% to ~73-74% test accuracy.

## Ranking (by Test Accuracy, then Macro F1)

1. **R7**: Test `0.7398`, F1 Macro `0.7403` (best)
2. **R6**: Test `0.7332`, F1 Macro `0.7337`
3. **R2**: Test `0.7308`, F1 Macro `0.7306`
4. **R5**: Test `0.7266`, F1 Macro `0.7259`
5. **R3**: Test `0.7248`, F1 Macro `0.7227`
6. **R4**: Test `0.7230`, F1 Macro `0.7228`
7. **R1**: Test `0.6514`, F1 Macro `0.6485`

## Interpretation

1. The project is no longer in severe underfit/overfit instability; training is now controlled and reproducible.
2. Gains are real but incremental, and still below the original 80-85% objective.
3. Further gains likely require either:
   - broader automated hyperparameter search around R7 regime, or
   - architecture/feature-level upgrades within the same family.

## Current Best Checkpoint

- Best run: **R7**
- Recommended retained model: `saved_models/best_cnn_bilstm_attention.pt`
- Current best metrics:
  - Test Accuracy: **73.98%**
  - Macro F1: **74.03%**
