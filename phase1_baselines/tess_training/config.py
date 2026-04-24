from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "TESS"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

SR = 16000
N_MFCC = 13
N_MELS = 128

CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)