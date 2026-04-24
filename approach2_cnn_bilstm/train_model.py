"""
Training Module
Handles model training with monitoring and checkpointing (PyTorch)
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    LR_REDUCE_MIN_LR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    TRAINING_PLOT_PATH,
    CHECKPOINT_DIR,
    SAVE_INTERVAL,
    LOG_LEVEL,
    GRADIENT_CLIP_NORM,
    SAVED_MODELS_DIR,
    USE_LR_SCHEDULER,
    LR_SCHEDULER_TYPE,
    COSINE_T_MAX,
    COSINE_ETA_MIN,
    MIN_EPOCHS_BEFORE_EARLY_STOP,
    LABEL_SMOOTHING,
)
from model_builder import build_training_components

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitors training progress and saves plots."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: Dict, save_path: Path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history.get('epoch', []), history.get('train_loss', []), label='Train Loss', linewidth=2)
        axes[0].plot(history.get('epoch', []), history.get('val_loss', []), label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.get('epoch', []), history.get('train_accuracy', []), label='Train Accuracy', linewidth=2)
        axes[1].plot(history.get('epoch', []), history.get('val_accuracy', []), label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training plot saved to {save_path}")


def _save_checkpoint(path: Path, checkpoint: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))


def _run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: Optional[torch.device] = None,
    scaler: Optional[object] = None,
    input_shape: Optional[tuple] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """Train a PyTorch SER model and save checkpoints/history."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion, optimizer = build_training_components(
        model,
        learning_rate=learning_rate,
        class_weights=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    scheduler = None
    if USE_LR_SCHEDULER:
        if LR_SCHEDULER_TYPE == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=COSINE_T_MAX,
                eta_min=COSINE_ETA_MIN,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=LR_REDUCE_FACTOR,
                patience=LR_REDUCE_PATIENCE,
                min_lr=LR_REDUCE_MIN_LR,
            )

    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0

    logger.info('=' * 70)
    logger.info('STARTING MODEL TRAINING (PYTORCH)')
    logger.info('=' * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_dataset, criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_dataset, criterion, optimizer, device, train=False)

        current_lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch)
        history['train_loss'].append(float(train_loss))
        history['train_accuracy'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_accuracy'].append(float(val_acc))
        history['learning_rate'].append(float(current_lr))

        if scheduler is not None:
            if LR_SCHEDULER_TYPE == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_acc)

        improved = val_acc > (best_val_acc + EARLY_STOPPING_MIN_DELTA)
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())

            best_checkpoint = {
                'model_state_dict': best_state,
                'input_shape': input_shape,
                'num_classes': model.classifier[-1].out_features,
                'best_val_accuracy': float(best_val_acc),
                'epoch': best_epoch,
                'scaler_mean': scaler.mean_.astype(np.float32).tolist() if scaler is not None else None,
                'scaler_scale': scaler.scale_.astype(np.float32).tolist() if scaler is not None else None,
            }
            _save_checkpoint(BEST_MODEL_PATH, best_checkpoint)
        else:
            patience_counter += 1

        if epoch % SAVE_INTERVAL == 0:
            periodic_checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_shape': input_shape,
                'epoch': epoch,
                'val_accuracy': float(val_acc),
            }
            _save_checkpoint(CHECKPOINT_DIR / f'model_epoch_{epoch:03d}.pt', periodic_checkpoint)

        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if epoch >= MIN_EPOCHS_BEFORE_EARLY_STOP and patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Restore best weights before final export
    if best_state is not None:
        model.load_state_dict(best_state)

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_shape': input_shape,
        'num_classes': model.classifier[-1].out_features,
        'best_val_accuracy': float(best_val_acc),
        'best_epoch': best_epoch,
        'scaler_mean': scaler.mean_.astype(np.float32).tolist() if scaler is not None else None,
        'scaler_scale': scaler.scale_.astype(np.float32).tolist() if scaler is not None else None,
        'history': history,
    }
    _save_checkpoint(FINAL_MODEL_PATH, final_checkpoint)
    logger.info(f"Final model saved to {FINAL_MODEL_PATH}")

    metrics_path = SAVED_MODELS_DIR / 'training_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {metrics_path}")

    monitor = TrainingMonitor(SAVED_MODELS_DIR)
    monitor.plot_training_history(history, TRAINING_PLOT_PATH)

    return history


def evaluate_training(history: Dict) -> Dict:
    summary = {
        'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else 0.0,
        'final_val_loss': float(history['val_loss'][-1]) if history['val_loss'] else 0.0,
        'final_train_accuracy': float(history['train_accuracy'][-1]) if history['train_accuracy'] else 0.0,
        'final_val_accuracy': float(history['val_accuracy'][-1]) if history['val_accuracy'] else 0.0,
        'best_val_accuracy': float(max(history['val_accuracy'])) if history['val_accuracy'] else 0.0,
        'best_val_accuracy_epoch': int(np.argmax(history['val_accuracy']) + 1) if history['val_accuracy'] else 0,
        'total_epochs': len(history['epoch']),
    }

    if history['train_accuracy'] and history['val_accuracy']:
        overfitting_gap = history['train_accuracy'][-1] - history['val_accuracy'][-1]
        summary['overfitting_gap'] = float(overfitting_gap)
        if overfitting_gap < 0.05:
            summary['overfitting_status'] = 'Good'
        elif overfitting_gap < 0.1:
            summary['overfitting_status'] = 'Moderate'
        else:
            summary['overfitting_status'] = 'High'

    return summary


if __name__ == '__main__':
    print('Training module loaded successfully!')
    print('Use this module with train_model function to train the SER model.')
