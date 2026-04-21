"""
Evaluation Module
Evaluates model performance and generates metrics
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    EMOTIONS, CONFUSION_MATRIX_PATH, METRICS_PATH,
    CLASSIFICATION_REPORT_PATH, LOG_LEVEL, IDX_TO_EMOTION
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance on test set"""
    
    def __init__(self, model: tf.keras.Model, class_names: list = EMOTIONS):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            class_names: List of emotion class names
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("="*70)
        logger.info("EVALUATING MODEL ON TEST SET")
        logger.info("="*70)
        
        # Get predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        logger.info("Making predictions...")
        for X_batch, y_batch in test_dataset:
            predictions = self.model.predict(X_batch, verbose=0)
            y_pred_proba.extend(predictions)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(y_batch.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        logger.info(f"Predictions made on {len(y_true)} samples")
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        
        # Basic accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class breakdown
        per_class_metrics = {}
        for idx, emotion in enumerate(self.class_names):
            mask = y_true == idx
            if mask.sum() > 0:
                class_acc = np.mean(y_true[mask] == y_pred[mask])
                class_f1 = f1_score(y_true == idx, y_pred == idx, zero_division=0)
                
                per_class_metrics[emotion] = {
                    'accuracy': float(class_acc),
                    'f1': float(class_f1),
                    'samples': int(mask.sum())
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'overall_accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics
        }
        
        logger.info("\n" + "="*70)
        logger.info("TEST SET METRICS")
        logger.info("="*70)
        logger.info(f"Overall Accuracy:      {accuracy:.4f}")
        logger.info(f"Precision (weighted):  {precision:.4f}")
        logger.info(f"Recall (weighted):     {recall:.4f}")
        logger.info(f"F1-Score (weighted):   {f1_weighted:.4f}")
        logger.info(f"F1-Score (macro):      {f1_macro:.4f}")
        
        logger.info("\nPer-Class Metrics:")
        logger.info("-" * 70)
        for emotion, metrics_dict in per_class_metrics.items():
            logger.info(f"{emotion:12s} | Acc: {metrics_dict['accuracy']:.4f} | "
                       f"F1: {metrics_dict['f1']:.4f} | Samples: {metrics_dict['samples']}")
        logger.info("="*70 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Path = CONFUSION_MATRIX_PATH):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Absolute Values)', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Normalized values
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, metrics: Dict, save_dir: Path = None):
        """Plot per-class metrics"""
        if save_dir is None:
            save_dir = Path(METRICS_PATH).parent
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        per_class = metrics['per_class_metrics']
        
        # Extract data
        emotions = list(per_class.keys())
        accuracies = [per_class[e]['accuracy'] for e in emotions]
        f1_scores = [per_class[e]['f1'] for e in emotions]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(len(emotions))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            ax.text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class metrics plot saved to {save_dir / 'per_class_metrics.png'}")
    
    def save_metrics_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                           metrics: Dict, save_path: Path = METRICS_PATH):
        """Save detailed metrics report"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names,
                                      output_dict=False)
        
        # Save text report
        report_path = CLASSIFICATION_REPORT_PATH
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(report)
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"Classification report saved to {report_path}")
        
        # Save metrics JSON
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {save_path}")


def generate_evaluation_report(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    output_dir: Path = None
) -> Dict:
    """
    Generate comprehensive evaluation report
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with evaluation results
    """
    if output_dir is None:
        output_dir = Path(METRICS_PATH).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(test_dataset)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    metrics = results['metrics']
    
    # Generate plots and reports
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_per_class_metrics(metrics, output_dir)
    evaluator.save_metrics_report(y_true, y_pred, metrics)
    
    logger.info(f"\nEvaluation report generated in {output_dir}")
    
    return results


if __name__ == '__main__':
    print("Evaluation module loaded successfully!")
    print("Use this module with ModelEvaluator class to evaluate trained models.")
