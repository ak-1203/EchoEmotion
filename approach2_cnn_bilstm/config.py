"""
SER Model Configuration
Centralized configuration for all hyperparameters and paths
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'/ 'combined_dataset'
SAVED_MODELS_DIR = PROJECT_ROOT / 'Approach2'/ 'saved_models'
RESULTS_DIR = PROJECT_ROOT / 'Approach2_cnn-bilstm' /'results'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories if they don't exist
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
SAMPLE_RATE = 16000  # Hz
N_MELS = 64  # Number of mel frequency bins
N_FFT = 1024  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames
N_MFCC = 13  # Number of MFCCs (if using MFCC)

# Audio preprocessing
MIN_DURATION = 0.5  # Minimum audio duration in seconds
MAX_DURATION = 10.0  # Maximum audio duration in seconds
TARGET_DURATION = 3.0  # Target audio duration for padding/trimming

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5  # Balanced augmentation (avoid over-regularization)

# SpecAugment parameters
SPEC_AUGMENT_FREQ_MASK_PARAM = 30  # F parameter for freq masking
SPEC_AUGMENT_TIME_MASK_PARAM = 40  # T parameter for time masking
SPEC_AUGMENT_NUM_MASKS = 2  # Number of mask regions to apply

# Time-domain augmentation
PITCH_SHIFT_RANGE = (-3, 3)  # Semitones (reserved)
TIME_SHIFT_RANGE = (-0.1, 0.1)  # Fraction of duration
NOISE_FACTOR = 0.005  # Gaussian noise standard deviation

# ============================================================================
# DATASET SPLIT
# ============================================================================
TRAIN_SPLIT = 0.70  # 70% training
VAL_SPLIT = 0.15  # 15% validation
TEST_SPLIT = 0.15  # 15% testing
RANDOM_SEED = 42

# ============================================================================
# EMOTION LABELS
# ============================================================================
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# CNN Parameters
CNN_FILTERS = [64, 128, 256]  # Filters for each CNN block
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.2

# BiLSTM Parameters
LSTM_UNITS = 256
LSTM_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.0  # Reserved for compatibility

# Attention Parameters
NUM_ATTENTION_HEADS = 4
ATTENTION_HEAD_DIM = 64

# Regularization
L2_REGULARIZATION = 0.0005
DROPOUT_RATE = 0.2
BATCH_NORM = True

# ============================================================================
# TRAINING
# ============================================================================
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.0004
NUM_WORKERS = 0
PIN_MEMORY = True

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'cosine'  # 'plateau' or 'cosine'
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 10
LR_REDUCE_MIN_LR = 1e-6
COSINE_T_MAX = 30
COSINE_ETA_MIN = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 0.0005
MIN_EPOCHS_BEFORE_EARLY_STOP = 60

# Optimizer
OPTIMIZER = 'adam'  # 'adam', 'adamw', 'rmsprop'
GRADIENT_CLIP_NORM = 1.0
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.05

# ============================================================================
# VALIDATION & TESTING
# ============================================================================
CALCULATE_METRICS = True
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_accuracy'
MONITOR_MODE = 'max'

# ============================================================================
# GPU/DEVICE
# ============================================================================
USE_GPU = True
ALLOW_GROWTH = True  # Kept for compatibility

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
VERBOSE = 1
LOG_LEVEL = 'INFO'
SAVE_INTERVAL = 5  # Save periodic checkpoint every N epochs
PLOT_INTERVAL = 1

# Model naming
MODEL_NAME = 'cnn_bilstm_attention'
MODEL_SUFFIX = '.pt'

# ============================================================================
# PATHS FOR TRAINED MODELS
# ============================================================================
BEST_MODEL_PATH = SAVED_MODELS_DIR / f'best_{MODEL_NAME}{MODEL_SUFFIX}'
FINAL_MODEL_PATH = SAVED_MODELS_DIR / f'final_{MODEL_NAME}{MODEL_SUFFIX}'
CHECKPOINT_DIR = SAVED_MODELS_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# OUTPUT PATHS
# ============================================================================
TRAINING_PLOT_PATH = RESULTS_DIR / 'training_history.png'
CONFUSION_MATRIX_PATH = RESULTS_DIR / 'confusion_matrix.png'
METRICS_PATH = RESULTS_DIR / 'metrics.json'
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / 'classification_report.txt'

# ============================================================================
# INFERENCE
# ============================================================================
INFERENCE_CONFIDENCE_THRESHOLD = 0.5
RETURN_TOP_K = 3

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
DETERMINISTIC = True


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SER MODEL CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Saved Models: {SAVED_MODELS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print(f"\nEmotions: {EMOTIONS}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"\nAudio: SR={SAMPLE_RATE}Hz, Mels={N_MELS}, FFT={N_FFT}")
    print(f"Train/Val/Test: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    print(f"\nModel: {MODEL_NAME}")
    print(f"CNN Filters: {CNN_FILTERS}")
    print(f"LSTM Units: {LSTM_UNITS}")
    print(f"Attention Heads: {NUM_ATTENTION_HEADS}")
    print(f"\nBatch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {EPOCHS}")
    print("=" * 70 + "\n")
