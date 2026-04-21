"""
Setup Script for Speech Emotion Recognition System
Handles environment setup, dependency installation, and validation
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    logger.info('Checking Python version...')
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logger.error(f'Python 3.9+ required. Found: {version.major}.{version.minor}')
        return False
    logger.info(f'OK Python {version.major}.{version.minor}.{version.micro}')
    return True


def check_gpu():
    logger.info('\nChecking GPU availability...')
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"OK Found {torch.cuda.device_count()} GPU device(s)")
            for idx in range(torch.cuda.device_count()):
                logger.info(f"  - {torch.cuda.get_device_name(idx)}")
            return True

        logger.warning('No GPU found, will use CPU (training will be slower)')
        return False
    except Exception as exc:
        logger.warning(f'Could not check GPU: {exc}')
        return False


def install_dependencies():
    logger.info('\nInstalling dependencies...')
    requirements_file = Path(__file__).parent / 'requirements.txt'

    if not requirements_file.exists():
        logger.error(f'requirements.txt not found at {requirements_file}')
        return False

    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)], check=True)
        logger.info('OK Dependencies installed successfully')
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f'Failed to install dependencies: {exc}')
        return False


def verify_imports():
    logger.info('\nVerifying dependencies...')
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
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
            logger.info(f'OK {name}')
        except ImportError:
            logger.error(f'FAIL {name} - not installed')
            all_ok = False

    return all_ok


def check_data_directory():
    logger.info('\nChecking dataset...')

    try:
        from config import DATA_DIR, EMOTIONS
    except Exception as exc:
        logger.error(f'Could not import config: {exc}')
        return False

    if not DATA_DIR.exists():
        logger.error(f'Data directory not found: {DATA_DIR}')
        return False

    logger.info(f'OK Data directory found: {DATA_DIR}')

    total_files = 0
    for emotion in EMOTIONS:
        emotion_dir = DATA_DIR / emotion
        if not emotion_dir.exists():
            logger.warning(f'Missing emotion directory: {emotion}')
            continue

        wav_files = list(emotion_dir.glob('*.wav'))
        logger.info(f'  {emotion}: {len(wav_files)} files')
        total_files += len(wav_files)

    if total_files == 0:
        logger.error('No audio files found in dataset')
        return False

    logger.info(f'OK Total audio files: {total_files}')
    return True


def create_directories():
    logger.info('\nCreating output directories...')

    try:
        from config import SAVED_MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINT_DIR

        for dir_path in [SAVED_MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f'OK {dir_path}')

        return True
    except Exception as exc:
        logger.error(f'Failed to create directories: {exc}')
        return False


def test_data_loading():
    logger.info('\nTesting data loading (sampling first 10 files)...')

    try:
        from config import DATA_DIR, EMOTIONS
        from data_processor import AudioProcessor

        processor = AudioProcessor()
        files_tested = 0

        for emotion in EMOTIONS:
            emotion_dir = DATA_DIR / emotion
            for wav_file in list(emotion_dir.glob('*.wav'))[:2]:
                y = processor.load_audio(str(wav_file))
                if y is None:
                    continue
                mel_spec = processor.extract_melspectrogram(y)
                _ = processor.pad_or_trim(mel_spec)
                files_tested += 1

        logger.info(f'OK Successfully processed {files_tested} files')
        return files_tested > 0
    except Exception as exc:
        logger.error(f'Data loading test failed: {exc}')
        return False


def print_summary(results):
    print('\n' + '=' * 70)
    print('SETUP SUMMARY')
    print('=' * 70)

    checks = [
        ('Python Version', results.get('python_version')),
        ('GPU Available', results.get('gpu_check')),
        ('Dependencies Installed', results.get('dependencies')),
        ('Dependencies Verified', results.get('verify_imports')),
        ('Dataset Found', results.get('data_directory')),
        ('Directories Created', results.get('create_directories')),
        ('Data Loading Test', results.get('data_loading')),
    ]

    for check_name, result in checks:
        status = 'OK' if result else 'FAIL'
        print(f"{check_name:.<50} {status}")

    print('=' * 70)


def main():
    print('\n' + '#' * 70)
    print('# SPEECH EMOTION RECOGNITION - SETUP WIZARD (PYTORCH)')
    print('#' * 70 + '\n')

    results = {}
    results['python_version'] = check_python_version()
    if not results['python_version']:
        sys.exit(1)

    results['gpu_check'] = check_gpu()
    results['dependencies'] = install_dependencies()
    results['verify_imports'] = verify_imports()
    results['data_directory'] = check_data_directory()
    results['create_directories'] = create_directories()
    results['data_loading'] = test_data_loading()

    print_summary(results)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nSetup cancelled by user.')
        sys.exit(1)
    except Exception as exc:
        logger.error(f'\nSetup failed with error: {exc}', exc_info=True)
        sys.exit(1)
