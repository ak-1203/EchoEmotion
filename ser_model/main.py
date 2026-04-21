"""
Main Orchestrator
Coordinates the complete SER pipeline: data loading, training, evaluation, and prediction (PyTorch)
"""

import argparse
import logging
import os
import random

import numpy as np
import torch

from config import (
    DATA_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LOG_LEVEL,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    USE_GPU,
    DETERMINISTIC,
    RANDOM_SEED, 
    N_MELS,
    TARGET_DURATION,
    SAMPLE_RATE,
    HOP_LENGTH,
    USE_CLASS_WEIGHTS,
    NUM_CLASSES,
)
from data_processor import DataLoader
from model_builder import build_ser_model
from train_model import train_model, evaluate_training
from evaluate_model import generate_evaluation_report
from predict import SERPredictor

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')
    return device


def setup_reproducibility():
    logger.info('Setting up reproducibility...')

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    logger.info(f'Random seed set to {RANDOM_SEED}')


def load_data(batch_size: int):
    logger.info('\n' + '=' * 70)
    logger.info('LOADING AND PREPARING DATA')
    logger.info('=' * 70)

    loader = DataLoader(DATA_DIR)
    data = loader.prepare_dataset()

    train_dataset = loader.create_torch_dataset(data['X_train'], data['y_train'], batch_size, augment=True, shuffle=True)
    val_dataset = loader.create_torch_dataset(data['X_val'], data['y_val'], batch_size, augment=False, shuffle=False)
    test_dataset = loader.create_torch_dataset(data['X_test'], data['y_test'], batch_size, augment=False, shuffle=False)

    logger.info('Data loaded successfully!')
    return train_dataset, val_dataset, test_dataset, data


def build_model(input_shape):
    logger.info('\n' + '=' * 70)
    logger.info('BUILDING MODEL')
    logger.info('=' * 70)
    return build_ser_model(input_shape)


def build_class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels."""
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.clip(class_counts, a_min=1.0, a_max=None)
    weights = class_counts.sum() / (NUM_CLASSES * class_counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training(model, train_dataset, val_dataset, epochs, learning_rate, device, scaler, input_shape, class_weights):
    logger.info('\n' + '=' * 70)
    logger.info('STARTING TRAINING')
    logger.info('=' * 70)

    history = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        scaler=scaler,
        input_shape=input_shape,
        class_weights=class_weights,
    )
    summary = evaluate_training(history)
    return model, summary


def run_evaluation(model, test_dataset, device):
    logger.info('\n' + '=' * 70)
    logger.info('STARTING EVALUATION')
    logger.info('=' * 70)
    return generate_evaluation_report(model, test_dataset, device=device)


def run_prediction(model_path, audio_paths, device):
    logger.info('\n' + '=' * 70)
    logger.info('MAKING PREDICTIONS')
    logger.info('=' * 70)

    predictor = SERPredictor(model_path, device=device)
    results = predictor.predict_batch(audio_paths)
    for result in results:
        if 'error' not in result:
            predictor.print_prediction(result)
    return results


def load_model_checkpoint(model_path, device):
    checkpoint = torch.load(str(model_path), map_location=device)
    input_shape = checkpoint.get('input_shape')
    if input_shape is None:
        input_shape = (1, N_MELS, int(TARGET_DURATION * SAMPLE_RATE // HOP_LENGTH))

    model = build_ser_model(tuple(input_shape)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def print_summary(train_summary, test_metrics):
    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)

    print('\nTraining Summary:')
    print('-' * 70)
    print(f"  Total Epochs:            {train_summary['total_epochs']}")
    print(f"  Final Train Loss:        {train_summary['final_train_loss']:.4f}")
    print(f"  Final Val Loss:          {train_summary['final_val_loss']:.4f}")
    print(f"  Final Train Accuracy:    {train_summary['final_train_accuracy']:.4f}")
    print(f"  Final Val Accuracy:      {train_summary['final_val_accuracy']:.4f}")
    print(
        f"  Best Val Accuracy:       {train_summary['best_val_accuracy']:.4f} "
        f"(epoch {train_summary['best_val_accuracy_epoch']})"
    )
    print(f"  Overfitting Gap:         {train_summary.get('overfitting_gap', 0.0):.4f}")
    print(f"  Overfitting Status:      {train_summary.get('overfitting_status', 'N/A')}")

    print('\nTest Set Performance:')
    print('-' * 70)
    print(f"  Overall Accuracy:        {test_metrics['overall_accuracy']:.4f}")
    print(f"  Precision (weighted):    {test_metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):       {test_metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted):     {test_metrics['f1_weighted']:.4f}")
    print(f"  F1-Score (macro):        {test_metrics['f1_macro']:.4f}")

    print('=' * 70 + '\n')


def print_evaluation_summary(metrics):
    print('\n' + '=' * 70)
    print('EVALUATION SUMMARY')
    print('=' * 70)

    print(f"\nOverall Accuracy:        {metrics['overall_accuracy']:.4f}")
    print(f"Precision (weighted):    {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):       {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"F1-Score (macro):        {metrics['f1_macro']:.4f}")

    print('\nPer-Class Metrics:')
    for emotion, em in metrics['per_class_metrics'].items():
        print(f"  {emotion:12s}: Acc={em['accuracy']:.4f}, F1={em['f1']:.4f}, Samples={em['samples']}")

    print('=' * 70 + '\n')


def main(args):
    logger.info('\n' + '#' * 70)
    logger.info('# SPEECH EMOTION RECOGNITION - COMPLETE PIPELINE (PYTORCH)')
    logger.info('#' * 70 + '\n')

    device = setup_device()
    setup_reproducibility()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f'Data directory not found: {DATA_DIR}')

    try:
        if args.mode == 'train':
            train_dataset, val_dataset, test_dataset, data = load_data(args.batch_size)

            input_shape = (1, data['X_train'].shape[1], data['X_train'].shape[2])
            logger.info(f'Input shape: {input_shape}')

            model = build_model(input_shape)
            class_weights = build_class_weights(data['y_train'], device) if USE_CLASS_WEIGHTS else None
            model, train_summary = run_training(
                model,
                train_dataset,
                val_dataset,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=device,
                scaler=data.get('scaler'),
                input_shape=input_shape,
                class_weights=class_weights,
            )

            test_results = run_evaluation(model, test_dataset, device)
            print_summary(train_summary, test_results['metrics'])

        elif args.mode == 'evaluate':
            model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(f'Model not found: {model_path}')

            _, _, test_dataset, _ = load_data(args.batch_size)
            model = load_model_checkpoint(model_path, device)
            test_results = run_evaluation(model, test_dataset, device)
            print_evaluation_summary(test_results['metrics'])

        elif args.mode == 'predict':
            if not args.audio_path:
                raise ValueError('--audio_path is required for predict mode')

            model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(f'No saved model found at {BEST_MODEL_PATH} or {FINAL_MODEL_PATH}')

            run_prediction(model_path, args.audio_path, device)

        else:
            raise ValueError(f'Unknown mode: {args.mode}')

    except Exception as exc:
        logger.error(f'Pipeline failed: {exc}', exc_info=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition - Complete Pipeline (PyTorch)')

    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], default='train', help='Pipeline mode')
    parser.add_argument('--audio_path', type=str, nargs='*', help='Path(s) to audio file(s) for prediction mode')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help=f'Learning rate (default: {LEARNING_RATE})')

    args = parser.parse_args()
    main(args)
