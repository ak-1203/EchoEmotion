"""
Main Orchestrator
Coordinates the complete SER pipeline: data loading, training, evaluation, and prediction
"""

import logging
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np

from config import (
    DATA_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, LOG_LEVEL,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, USE_GPU, GPU_MEMORY_FRACTION,
    ALLOW_GROWTH, DETERMINISTIC, TF_DETERMINISTIC_OPS, RANDOM_SEED
)
from data_processor import DataLoader
from model_builder import build_ser_model, compile_model
from train_model import train_model, evaluate_training
from evaluate_model import generate_evaluation_report
from predict import SERPredictor

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU settings"""
    logger.info("Configuring GPU...")
    
    if not USE_GPU:
        # Disable GPU
        tf.config.set_visible_devices([], 'GPU')
        logger.info("GPU disabled")
    else:
        # Configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    if ALLOW_GROWTH:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    else:
                        # Set fixed memory usage
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=int(tf.config.experimental.get_device_details(gpu)
                                               ['device_memory']
                                               * GPU_MEMORY_FRACTION)
                            )]
                        )
                logger.info(f"GPU configured with {len(gpus)} device(s)")
            except RuntimeError as e:
                logger.error(f"GPU configuration failed: {e}")
        else:
            logger.warning("No GPU found, using CPU")


def setup_reproducibility():
    """Set seeds for reproducibility"""
    logger.info("Setting up reproducibility...")
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    if DETERMINISTIC:
        tf.config.run_functions_eagerly(True)
        if TF_DETERMINISTIC_OPS:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    logger.info(f"Random seed set to {RANDOM_SEED}")


def load_data():
    """Load and prepare dataset"""
    logger.info("\n" + "="*70)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("="*70)
    
    loader = DataLoader(DATA_DIR)
    data = loader.prepare_dataset()
    
    # Create TF datasets
    train_dataset = loader.create_tf_dataset(
        data['X_train'], data['y_train'], BATCH_SIZE, augment=True
    )
    val_dataset = loader.create_tf_dataset(
        data['X_val'], data['y_val'], BATCH_SIZE, augment=False
    )
    test_dataset = loader.create_tf_dataset(
        data['X_test'], data['y_test'], BATCH_SIZE, augment=False
    )
    
    logger.info("Data loaded successfully!")
    
    return train_dataset, val_dataset, test_dataset, data


def build_model(input_shape):
    """Build and compile model"""
    logger.info("\n" + "="*70)
    logger.info("BUILDING MODEL")
    logger.info("="*70)
    
    model = build_ser_model(input_shape)
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    return model


def run_training(model, train_dataset, val_dataset):
    """Train the model"""
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    history = train_model(model, train_dataset, val_dataset, epochs=EPOCHS)
    summary = evaluate_training(history)
    
    return model, summary


def run_evaluation(model, test_dataset):
    """Evaluate model on test set"""
    logger.info("\n" + "="*70)
    logger.info("STARTING EVALUATION")
    logger.info("="*70)
    
    results = generate_evaluation_report(model, test_dataset)
    
    return results


def run_prediction(model_path, audio_paths):
    """Make predictions on audio files"""
    logger.info("\n" + "="*70)
    logger.info("MAKING PREDICTIONS")
    logger.info("="*70)
    
    predictor = SERPredictor(model_path)
    
    if isinstance(audio_paths, str):
        result = predictor.predict_single(audio_paths)
        predictor.print_prediction(result)
        return result
    else:
        results = predictor.predict_batch(audio_paths)
        for result in results:
            if 'error' not in result:
                predictor.print_prediction(result)
        return results


def main(args):
    """Main orchestration function"""
    
    logger.info("\n" + "#"*70)
    logger.info("# SPEECH EMOTION RECOGNITION - COMPLETE PIPELINE")
    logger.info("#"*70 + "\n")
    
    # Setup environment
    setup_gpu()
    setup_reproducibility()
    
    # Check data directory
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    try:
        if args.mode == 'train':
            # Training pipeline
            train_dataset, val_dataset, test_dataset, data = load_data()
            
            # Determine input shape
            input_shape = (data['X_train'].shape[1], data['X_train'].shape[2], 1)
            logger.info(f"Input shape: {input_shape}")
            
            # Build model
            model = build_model(input_shape)
            
            # Train
            model, train_summary = run_training(model, train_dataset, val_dataset)
            
            # Evaluate
            test_results = run_evaluation(model, test_dataset)
            
            logger.info("\n" + "#"*70)
            logger.info("# TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("#"*70 + "\n")
            
            # Print final summary
            print_summary(train_summary, test_results['metrics'])
        
        elif args.mode == 'evaluate':
            # Evaluation only (load existing model)
            logger.info(f"Loading model from {BEST_MODEL_PATH}")
            
            if not BEST_MODEL_PATH.exists():
                logger.error(f"Model not found: {BEST_MODEL_PATH}")
                raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")
            
            _, _, test_dataset, _ = load_data()
            
            from model_builder import MultiHeadAttention
            model = tf.keras.models.load_model(
                str(BEST_MODEL_PATH),
                custom_objects={'MultiHeadAttention': MultiHeadAttention}
            )
            
            test_results = run_evaluation(model, test_dataset)
            print_evaluation_summary(test_results['metrics'])
        
        elif args.mode == 'predict':
            # Inference on audio files
            if not args.audio_path:
                logger.error("--audio_path required for predict mode")
                return
            
            audio_paths = args.audio_path if isinstance(args.audio_path, list) else [args.audio_path]
            
            if not BEST_MODEL_PATH.exists():
                logger.warning(f"Model not found: {BEST_MODEL_PATH}, using final model")
                model_path = FINAL_MODEL_PATH
            else:
                model_path = BEST_MODEL_PATH
            
            run_prediction(model_path, audio_paths)
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def print_summary(train_summary, test_metrics):
    """Print training and evaluation summary"""
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nTraining Summary:")
    print("-" * 70)
    print(f"  Total Epochs:            {train_summary['total_epochs']}")
    print(f"  Final Train Loss:        {train_summary['final_train_loss']:.4f}")
    print(f"  Final Val Loss:          {train_summary['final_val_loss']:.4f}")
    print(f"  Final Train Accuracy:    {train_summary['final_train_accuracy']:.4f}")
    print(f"  Final Val Accuracy:      {train_summary['final_val_accuracy']:.4f}")
    print(f"  Best Val Accuracy:       {train_summary['best_val_accuracy']:.4f} "
          f"(epoch {train_summary['best_val_accuracy_epoch']})")
    print(f"  Overfitting Gap:         {train_summary['overfitting_gap']:.4f}")
    print(f"  Overfitting Status:      {train_summary['overfitting_status']}")
    
    print("\nTest Set Performance:")
    print("-" * 70)
    print(f"  Overall Accuracy:        {test_metrics['overall_accuracy']:.4f}")
    print(f"  Precision (weighted):    {test_metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):       {test_metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted):     {test_metrics['f1_weighted']:.4f}")
    print(f"  F1-Score (macro):        {test_metrics['f1_macro']:.4f}")
    
    print("="*70 + "\n")


def print_evaluation_summary(metrics):
    """Print evaluation summary"""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nOverall Accuracy:        {metrics['overall_accuracy']:.4f}")
    print(f"Precision (weighted):    {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):       {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"F1-Score (macro):        {metrics['f1_macro']:.4f}")
    
    print("\nPer-Class Metrics:")
    for emotion, em in metrics['per_class_metrics'].items():
        print(f"  {emotion:12s}: Acc={em['accuracy']:.4f}, F1={em['f1']:.4f}, "
              f"Samples={em['samples']}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    import os
    
    parser = argparse.ArgumentParser(
        description='Speech Emotion Recognition - Complete Pipeline'
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'predict'],
        default='train',
        help='Pipeline mode: train (full training + evaluation), evaluate (test only), '
             'or predict (inference)'
    )
    
    parser.add_argument(
        '--audio_path',
        type=str,
        nargs='*',
        help='Path(s) to audio file(s) for prediction mode'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.epochs != EPOCHS:
        import config as cfg
        cfg.EPOCHS = args.epochs
    
    main(args)
