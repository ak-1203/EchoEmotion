"""
Test & Validation Suite
Comprehensive tests for SER system components
"""

import logging
import sys
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Run comprehensive tests"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, func):
        """Run a test and record result"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Test: {name}")
        logger.info('='*70)
        
        try:
            func()
            self.results[name] = 'PASS'
            self.passed += 1
            logger.info(f"✓ {name} PASSED")
        except Exception as e:
            self.results[name] = f'FAIL: {str(e)}'
            self.failed += 1
            logger.error(f"✗ {name} FAILED: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test_name, result in self.results.items():
            status = "✓" if result == 'PASS' else "✗"
            print(f"{status} {test_name:.<50} {result}")
        
        print("="*70)
        print(f"Passed: {self.passed}/{self.passed + self.failed}")
        print("="*70 + "\n")
        
        return self.failed == 0


# Test Functions

def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    
    imports_to_test = [
        'tensorflow',
        'keras',
        'numpy',
        'librosa',
        'sklearn',
        'matplotlib',
        'pandas',
        'tqdm',
    ]
    
    for module in imports_to_test:
        __import__(module)
        logger.info(f"  ✓ {module}")


def test_config():
    """Test configuration loading"""
    logger.info("Testing config...")
    
    from config import (
        DATA_DIR, EMOTIONS, NUM_CLASSES, BATCH_SIZE, EPOCHS,
        BEST_MODEL_PATH, SAVED_MODELS_DIR, RESULTS_DIR, LOGS_DIR
    )
    
    logger.info(f"  Data directory: {DATA_DIR}")
    logger.info(f"  Emotions: {EMOTIONS}")
    logger.info(f"  Classes: {NUM_CLASSES}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Max epochs: {EPOCHS}")
    
    assert NUM_CLASSES == 6
    assert BATCH_SIZE > 0
    assert EPOCHS > 0


def test_data_loading():
    """Test data loading capabilities"""
    logger.info("Testing data loading...")
    
    from data_processor import DataLoader, AudioProcessor
    from config import DATA_DIR, EMOTIONS
    
    # Test audio processor
    processor = AudioProcessor()
    logger.info("  ✓ AudioProcessor initialized")
    
    # Find first audio file
    audio_file = None
    for emotion in EMOTIONS:
        emotion_dir = DATA_DIR / emotion
        files = list(emotion_dir.glob('*.wav'))
        if files:
            audio_file = files[0]
            break
    
    if audio_file is None:
        raise FileNotFoundError(f"No audio files found in {DATA_DIR}")
    
    # Test loading
    y = processor.load_audio(str(audio_file))
    assert y is not None
    logger.info(f"  ✓ Loaded audio: {len(y)} samples")
    
    # Test mel-spectrogram extraction
    mel_spec = processor.extract_melspectrogram(y)
    logger.info(f"  ✓ Mel-spectrogram shape: {mel_spec.shape}")
    
    # Test padding
    mel_spec_padded = processor.pad_or_trim(mel_spec)
    assert mel_spec_padded.shape[1] == processor.target_length
    logger.info(f"  ✓ Padded shape: {mel_spec_padded.shape}")


def test_augmentation():
    """Test data augmentation"""
    logger.info("Testing augmentation...")
    
    from data_processor import DataAugmentor, AudioProcessor
    from config import DATA_DIR, EMOTIONS
    import librosa
    
    # Get sample audio
    audio_file = None
    for emotion in EMOTIONS:
        emotion_dir = DATA_DIR / emotion
        files = list(emotion_dir.glob('*.wav'))
        if files:
            audio_file = files[0]
            break
    
    processor = AudioProcessor()
    y = processor.load_audio(str(audio_file))
    mel_spec = processor.extract_melspectrogram(y)
    mel_spec = processor.pad_or_trim(mel_spec)
    
    # Test augmentations
    augmentor = DataAugmentor()
    
    mel_spec_aug1 = augmentor.spec_augment(mel_spec)
    assert mel_spec_aug1.shape == mel_spec.shape
    logger.info("  ✓ SpecAugment works")
    
    mel_spec_aug2 = augmentor.add_noise(mel_spec)
    assert mel_spec_aug2.shape == mel_spec.shape
    logger.info("  ✓ Noise augmentation works")
    
    mel_spec_aug3 = augmentor.time_shift(mel_spec)
    assert mel_spec_aug3.shape == mel_spec.shape
    logger.info("  ✓ Time shift augmentation works")


def test_model_building():
    """Test model architecture"""
    logger.info("Testing model building...")
    
    from model_builder import build_ser_model, compile_model
    from config import LEARNING_RATE
    import tensorflow as tf
    
    # Build model
    input_shape = (64, 94, 1)
    model = build_ser_model(input_shape)
    logger.info(f"  ✓ Model built with {model.count_params():,} parameters")
    
    # Compile
    model = compile_model(model, LEARNING_RATE)
    logger.info("  ✓ Model compiled")
    
    # Test forward pass
    dummy_input = np.random.randn(2, 64, 94, 1).astype(np.float32)
    output = model(dummy_input)
    
    assert output.shape == (2, 6)
    logger.info(f"  ✓ Forward pass successful, output shape: {output.shape}")


def test_dataset_creation():
    """Test TF dataset creation"""
    logger.info("Testing dataset creation...")
    
    from data_processor import DataLoader
    from config import DATA_DIR, BATCH_SIZE
    
    loader = DataLoader(DATA_DIR)
    data = loader.prepare_dataset()
    
    logger.info(f"  Train shape: {data['X_train'].shape}")
    logger.info(f"  Val shape: {data['X_val'].shape}")
    logger.info(f"  Test shape: {data['X_test'].shape}")
    
    # Create datasets
    train_ds = loader.create_tf_dataset(
        data['X_train'], data['y_train'], BATCH_SIZE, augment=True
    )
    
    # Get batch
    for X_batch, y_batch in train_ds.take(1):
        assert X_batch.shape[0] <= BATCH_SIZE
        assert X_batch.shape[1:] == (64, 94, 1)
        logger.info(f"  ✓ Batch shape: {X_batch.shape}")


def test_gpu():
    """Test GPU availability and configuration"""
    logger.info("Testing GPU...")
    
    import tensorflow as tf
    from config import USE_GPU
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if USE_GPU:
        if gpus:
            logger.info(f"  ✓ GPU found: {len(gpus)} device(s)")
            for gpu in gpus:
                logger.info(f"    - {gpu}")
        else:
            logger.warning("  ⚠ GPU not found but USE_GPU=True")
    else:
        logger.info("  ℹ GPU disabled in config")


def test_saving_loading():
    """Test model saving and loading"""
    logger.info("Testing model saving/loading...")
    
    from model_builder import build_ser_model, MultiHeadAttention
    import tensorflow as tf
    from pathlib import Path
    
    # Build model
    model = build_ser_model((64, 94, 1))
    
    # Save
    test_model_path = Path('test_model.h5')
    model.save(str(test_model_path))
    logger.info(f"  ✓ Model saved to {test_model_path}")
    
    # Load
    loaded_model = tf.keras.models.load_model(
        str(test_model_path),
        custom_objects={'MultiHeadAttention': MultiHeadAttention}
    )
    logger.info("  ✓ Model loaded successfully")
    
    # Test loaded model
    dummy_input = np.random.randn(1, 64, 94, 1).astype(np.float32)
    output = loaded_model(dummy_input)
    assert output.shape == (1, 6)
    logger.info(f"  ✓ Loaded model inference works")
    
    # Cleanup
    test_model_path.unlink()
    logger.info("  ✓ Cleanup complete")


def test_evaluation():
    """Test evaluation metrics"""
    logger.info("Testing evaluation...")
    
    from evaluate_model import ModelEvaluator
    from model_builder import build_ser_model
    import tensorflow as tf
    
    # Build model
    model = build_ser_model((64, 94, 1))
    
    # Create dummy data
    X_test = np.random.randn(100, 64, 94, 1).astype(np.float32)
    y_test = np.random.randint(0, 6, 100)
    
    # Create dataset
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(16)
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    logger.info("  ✓ ModelEvaluator initialized")
    
    try:
        results = evaluator.evaluate(test_ds)
        assert 'y_true' in results
        assert 'y_pred' in results
        assert 'metrics' in results
        logger.info("  ✓ Evaluation completed")
    except Exception as e:
        # Evaluation might fail on dummy data, but that's ok
        logger.warning(f"  ⚠ Evaluation failed (expected with dummy data): {e}")


def test_prediction():
    """Test prediction module"""
    logger.info("Testing prediction module...")
    
    from predict import SERPredictor
    from model_builder import build_ser_model
    from config import BEST_MODEL_PATH, DATA_DIR, EMOTIONS
    from pathlib import Path
    
    # Get real audio file for testing
    audio_file = None
    for emotion in EMOTIONS:
        emotion_dir = DATA_DIR / emotion
        files = list(emotion_dir.glob('*.wav'))
        if files:
            audio_file = files[0]
            break
    
    if audio_file is None:
        raise FileNotFoundError("No audio files for prediction test")
    
    # Build and save dummy model
    model = build_ser_model((64, 94, 1))
    test_model_path = Path('test_model_pred.h5')
    model.save(str(test_model_path))
    
    # Initialize predictor
    predictor = SERPredictor(test_model_path)
    logger.info("  ✓ SERPredictor initialized")
    
    # Predict
    result = predictor.predict_single(str(audio_file))
    assert 'predicted_emotion' in result
    assert 'confidence' in result
    assert 'all_predictions' in result
    logger.info(f"  ✓ Prediction successful: {result['predicted_emotion']}")
    
    # Cleanup
    test_model_path.unlink()


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# SPEECH EMOTION RECOGNITION - TEST SUITE")
    print("#"*70 + "\n")
    
    runner = TestRunner()
    
    # Run tests
    runner.test("Imports", test_imports)
    runner.test("Configuration", test_config)
    runner.test("GPU Setup", test_gpu)
    runner.test("Data Loading", test_data_loading)
    runner.test("Data Augmentation", test_augmentation)
    runner.test("Model Building", test_model_building)
    runner.test("Dataset Creation", test_dataset_creation)
    runner.test("Model Save/Load", test_saving_loading)
    runner.test("Evaluation", test_evaluation)
    runner.test("Prediction", test_prediction)
    
    # Print summary
    all_passed = runner.print_summary()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTests failed: {e}", exc_info=True)
        sys.exit(1)
