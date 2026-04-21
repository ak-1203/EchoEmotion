"""
Setup Script for Speech Emotion Recognition System
Handles environment setup, dependency installation, and validation
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Verify Python 3.8+"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu():
    """Check GPU availability"""
    logger.info("\nChecking GPU availability...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✓ Found {len(gpus)} GPU device(s)")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
            return True
        else:
            logger.warning("⚠ No GPU found, will use CPU (training will be slower)")
            return False
    except Exception as e:
        logger.warning(f"⚠ Could not check GPU: {e}")
        return False


def install_dependencies():
    """Install Python packages from requirements.txt"""
    logger.info("\nInstalling dependencies...")
    
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        logger.error(f"requirements.txt not found at {requirements_file}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
            check=True
        )
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def verify_imports():
    """Verify key dependencies can be imported"""
    logger.info("\nVerifying dependencies...")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'librosa': 'Librosa',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - not installed")
            all_ok = False
    
    return all_ok


def check_data_directory():
    """Verify combined_dataset structure"""
    logger.info("\nChecking dataset...")
    
    try:
        from config import DATA_DIR, EMOTIONS
    except Exception as e:
        logger.error(f"Could not import config: {e}")
        return False
    
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return False
    
    logger.info(f"✓ Data directory found: {DATA_DIR}")
    
    # Check emotion subdirectories
    total_files = 0
    for emotion in EMOTIONS:
        emotion_dir = DATA_DIR / emotion
        if not emotion_dir.exists():
            logger.warning(f"⚠ Emotion directory not found: {emotion}")
            continue
        
        wav_files = list(emotion_dir.glob('*.wav'))
        logger.info(f"  {emotion}: {len(wav_files)} files")
        total_files += len(wav_files)
    
    if total_files == 0:
        logger.error("No audio files found in dataset!")
        return False
    
    logger.info(f"✓ Total audio files: {total_files}")
    return True


def create_directories():
    """Create necessary output directories"""
    logger.info("\nCreating output directories...")
    
    try:
        from config import SAVED_MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINT_DIR
        
        dirs = [SAVED_MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINT_DIR]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ {dir_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False


def test_data_loading():
    """Quick test of data loading"""
    logger.info("\nTesting data loading (sampling first 10 files)...")
    
    try:
        from config import DATA_DIR, EMOTIONS
        from data_processor import AudioProcessor
        from pathlib import Path
        
        processor = AudioProcessor()
        files_tested = 0
        files_failed = 0
        
        for emotion in EMOTIONS:
            emotion_dir = DATA_DIR / emotion
            wav_files = list(emotion_dir.glob('*.wav'))[:2]  # Test first 2 of each
            
            for wav_file in wav_files:
                try:
                    y = processor.load_audio(str(wav_file))
                    if y is not None:
                        mel_spec = processor.extract_melspectrogram(y)
                        mel_spec = processor.pad_or_trim(mel_spec)
                        files_tested += 1
                except Exception as e:
                    logger.warning(f"⚠ Failed to process {wav_file.name}: {e}")
                    files_failed += 1
        
        logger.info(f"✓ Successfully processed {files_tested} files")
        if files_failed > 0:
            logger.warning(f"⚠ Failed to process {files_failed} files")
        
        return files_tested > 0
    
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False


def setup_gpu_memory():
    """Optimize GPU memory settings"""
    logger.info("\nOptimizing GPU settings...")
    
    try:
        import tensorflow as tf
        from config import ALLOW_GROWTH, GPU_MEMORY_FRACTION
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                if ALLOW_GROWTH:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✓ GPU memory optimization enabled")
            return True
        else:
            logger.info("No GPU found, skipping GPU optimization")
            return True
    
    except Exception as e:
        logger.warning(f"⚠ Could not optimize GPU: {e}")
        return True  # Not critical


def print_summary(results):
    """Print setup summary"""
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    checks = [
        ("Python Version", results.get('python_version')),
        ("GPU Available", results.get('gpu_check')),
        ("Dependencies Installed", results.get('dependencies')),
        ("Dependencies Verified", results.get('verify_imports')),
        ("Dataset Found", results.get('data_directory')),
        ("Directories Created", results.get('create_directories')),
        ("Data Loading Test", results.get('data_loading')),
        ("GPU Memory Optimization", results.get('gpu_memory')),
    ]
    
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{check_name:.<50} {status}")
    
    all_passed = all(v for v in results.values() if v is not None)
    
    print("="*70)
    
    if all_passed:
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. python main.py --mode train")
        print("  2. Check results in results/ directory")
        print("  3. python main.py --mode predict --audio_path <audio.wav>")
        print("\nFor detailed usage: see USAGE_GUIDE.md")
    else:
        print("\n⚠ Setup completed with warnings.")
        print("Please fix the issues above before running training.")
    
    print("\n")


def main():
    """Run complete setup"""
    print("\n" + "#"*70)
    print("# SPEECH EMOTION RECOGNITION - SETUP WIZARD")
    print("#"*70 + "\n")
    
    results = {}
    
    # Run checks
    results['python_version'] = check_python_version()
    if not results['python_version']:
        print("\nSetup failed: Python 3.8+ required")
        sys.exit(1)
    
    results['gpu_check'] = check_gpu()
    results['dependencies'] = install_dependencies()
    results['verify_imports'] = verify_imports()
    results['data_directory'] = check_data_directory()
    results['create_directories'] = create_directories()
    results['data_loading'] = test_data_loading()
    results['gpu_memory'] = setup_gpu_memory()
    
    # Print summary
    print_summary(results)
    
    # Exit code based on results
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nSetup failed with error: {e}", exc_info=True)
        sys.exit(1)
