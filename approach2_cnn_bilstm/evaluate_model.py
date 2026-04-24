"""
Evaluation Module
Evaluates model performance and generates metrics (PyTorch)
"""

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from config import EMOTIONS, CONFUSION_MATRIX_PATH, METRICS_PATH, CLASSIFICATION_REPORT_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance on test set."""

    def __init__(self, model: torch.nn.Module, class_names: list = EMOTIONS, device: torch.device = None):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate(self, test_dataset) -> Dict:
        logger.info('=' * 70)
        logger.info('EVALUATING MODEL ON TEST SET')
        logger.info('=' * 70)

        self.model.eval()
        y_true, y_pred, y_pred_proba = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in test_dataset:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_pred_proba.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        metrics = self._calculate_metrics(y_true, y_pred)
        return {'y_true': y_true, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'metrics': metrics}

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        accuracy = float(np.mean(y_true == y_pred))
        precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        per_class_metrics = {}
        for idx, emotion in enumerate(self.class_names):
            mask = y_true == idx
            if mask.sum() > 0:
                class_acc = float(np.mean(y_true[mask] == y_pred[mask]))
                class_f1 = float(f1_score(y_true == idx, y_pred == idx, zero_division=0))
                per_class_metrics[emotion] = {
                    'accuracy': class_acc,
                    'f1': class_f1,
                    'samples': int(mask.sum()),
                }

        cm = confusion_matrix(y_true, y_pred)

        return {
            'overall_accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
        }

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = CONFUSION_MATRIX_PATH):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Absolute Values)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def plot_per_class_metrics(self, metrics: Dict, save_dir: Path = None):
        if save_dir is None:
            save_dir = Path(METRICS_PATH).parent

        save_dir.mkdir(parents=True, exist_ok=True)
        per_class = metrics['per_class_metrics']

        emotions = list(per_class.keys())
        accuracies = [per_class[e]['accuracy'] for e in emotions]
        f1_scores = [per_class[e]['f1'] for e in emotions]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(emotions))
        width = 0.35

        ax.bar(x - width / 2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width / 2, f1_scores, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Emotion')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_metrics_report(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict, save_path: Path = METRICS_PATH):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=False)

        with open(CLASSIFICATION_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write('=' * 70 + '\n')
            f.write('CLASSIFICATION REPORT\n')
            f.write('=' * 70 + '\n\n')
            f.write(report)
            f.write('\n' + '=' * 70 + '\n')

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)


def generate_evaluation_report(model: torch.nn.Module, test_dataset, output_dir: Path = None, device: torch.device = None) -> Dict:
    if output_dir is None:
        output_dir = Path(METRICS_PATH).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(model, device=device)
    results = evaluator.evaluate(test_dataset)

    y_true = results['y_true']
    y_pred = results['y_pred']
    metrics = results['metrics']

    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_per_class_metrics(metrics, output_dir)
    evaluator.save_metrics_report(y_true, y_pred, metrics)

    logger.info(f"\nEvaluation report generated in {output_dir}")
    return results


if __name__ == '__main__':
    print('Evaluation module loaded successfully!')
    print('Use this module with ModelEvaluator class to evaluate trained models.')
