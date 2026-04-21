"""
Training Module
Handles model training with callbacks, monitoring, and checkpointing
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, LR_REDUCE_MIN_LR,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, TRAINING_PLOT_PATH,
    CHECKPOINT_DIR, SAVE_INTERVAL, VERBOSE, MONITOR_METRIC,
    MONITOR_MODE, LOG_LEVEL, GRADIENT_CLIP_NORM, SAVED_MODELS_DIR
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class CustomMetricsCallback(callbacks.Callback):
    """Custom callback to track additional metrics during training"""
    
    def __init__(self, val_dataset, log_dir: Path):
        """
        Initialize callback
        
        Args:
            val_dataset: Validation dataset
            log_dir: Directory to save logs
        """
        super().__init__()
        self.val_dataset = val_dataset
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """Save metrics at epoch end"""
        if logs is not None:
            self.metrics_history['epoch'].append(epoch + 1)
            self.metrics_history['train_loss'].append(float(logs.get('loss', 0)))
            self.metrics_history['train_accuracy'].append(float(logs.get('accuracy', 0)))
            self.metrics_history['val_loss'].append(float(logs.get('val_loss', 0)))
            self.metrics_history['val_accuracy'].append(float(logs.get('val_accuracy', 0)))
            self.metrics_history['learning_rate'].append(
                float(self.model.optimizer.learning_rate.numpy())
            )
    
    def save_history(self, filepath):
        """Save metrics history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")


class TrainingMonitor:
    """Monitors training progress and saves plots"""
    
    def __init__(self, save_dir: Path):
        """Initialize monitor"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, history: Dict, save_path: Path):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(history.get('epoch', []), history.get('train_loss', []),
                    label='Train Loss', linewidth=2)
        axes[0].plot(history.get('epoch', []), history.get('val_loss', []),
                    label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history.get('epoch', []), history.get('train_accuracy', []),
                    label='Train Accuracy', linewidth=2)
        axes[1].plot(history.get('epoch', []), history.get('val_accuracy', []),
                    label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plot saved to {save_path}")


def get_callbacks(val_dataset) -> list:
    """
    Create training callbacks
    
    Args:
        val_dataset: Validation dataset
        
    Returns:
        List of callbacks
    """
    callback_list = []
    
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor=MONITOR_METRIC,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        min_delta=0.0001,
        verbose=1
    )
    callback_list.append(early_stopping)
    
    # Learning rate reduction
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor=MONITOR_METRIC,
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=LR_REDUCE_MIN_LR,
        verbose=1
    )
    callback_list.append(reduce_lr)
    
    # Model checkpointing (best model)
    checkpoint_best = callbacks.ModelCheckpoint(
        str(BEST_MODEL_PATH),
        monitor=MONITOR_METRIC,
        save_best_only=True,
        verbose=0
    )
    callback_list.append(checkpoint_best)
    
    # Periodic checkpoint
    checkpoint_periodic = callbacks.ModelCheckpoint(
        str(CHECKPOINT_DIR / 'model_epoch_{epoch:03d}.h5'),
        save_freq=SAVE_INTERVAL * 100,  # Save every SAVE_INTERVAL batches
        verbose=0
    )
    callback_list.append(checkpoint_periodic)
    
    # TensorBoard logging
    tensorboard = callbacks.TensorBoard(
        log_dir=str(SAVED_MODELS_DIR / 'tensorboard_logs'),
        histogram_freq=1,
        update_freq='epoch'
    )
    callback_list.append(tensorboard)
    
    logger.info(f"Created {len(callback_list)} callbacks")
    
    return callback_list


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = EPOCHS
) -> Dict:
    """
    Train the model
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs
        
    Returns:
        Training history dictionary
    """
    logger.info("="*70)
    logger.info("STARTING MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    
    # Get callbacks
    callback_list = get_callbacks(val_dataset)
    metrics_callback = CustomMetricsCallback(val_dataset, SAVED_MODELS_DIR / 'metrics')
    callback_list.append(metrics_callback)
    
    # Train model
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callback_list,
            verbose=VERBOSE
        )
        
        logger.info("="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        # Save final model
        model.save(str(FINAL_MODEL_PATH))
        logger.info(f"Final model saved to {FINAL_MODEL_PATH}")
        
        # Save metrics history
        metrics_path = SAVED_MODELS_DIR / 'training_metrics.json'
        metrics_callback.save_history(metrics_path)
        
        # Plot training history
        monitor = TrainingMonitor(SAVED_MODELS_DIR)
        monitor.plot_training_history(metrics_callback.metrics_history, TRAINING_PLOT_PATH)
        
        return metrics_callback.metrics_history
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        model.save(str(FINAL_MODEL_PATH))
        logger.info(f"Model saved to {FINAL_MODEL_PATH}")
        return metrics_callback.metrics_history
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def evaluate_training(history: Dict) -> Dict:
    """
    Evaluate training results
    
    Args:
        history: Training history dictionary
        
    Returns:
        Summary statistics
    """
    summary = {
        'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else 0,
        'final_val_loss': float(history['val_loss'][-1]) if history['val_loss'] else 0,
        'final_train_accuracy': float(history['train_accuracy'][-1]) if history['train_accuracy'] else 0,
        'final_val_accuracy': float(history['val_accuracy'][-1]) if history['val_accuracy'] else 0,
        'best_val_accuracy': float(max(history['val_accuracy'])) if history['val_accuracy'] else 0,
        'best_val_accuracy_epoch': int(np.argmax(history['val_accuracy']) + 1) if history['val_accuracy'] else 0,
        'total_epochs': len(history['epoch']),
    }
    
    # Calculate overfitting metric
    if history['train_accuracy'] and history['val_accuracy']:
        train_final = history['train_accuracy'][-1]
        val_final = history['val_accuracy'][-1]
        overfitting_gap = train_final - val_final
        summary['overfitting_gap'] = float(overfitting_gap)
        summary['overfitting_status'] = 'Good' if overfitting_gap < 0.05 else 'Moderate' if overfitting_gap < 0.1 else 'High'
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"{key:.<40} {value:.4f}")
        else:
            logger.info(f"{key:.<40} {value}")
    
    logger.info("="*70 + "\n")
    
    return summary


if __name__ == '__main__':
    print("Training module loaded successfully!")
    print("Use this module with train_model function to train the SER model.")
