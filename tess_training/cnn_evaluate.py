import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR, RESULTS_DIR, ensure_dirs
from data_utils import get_speaker_split_indices, load_tess_dataset
from gpu_utils import configure_gpu


def load_model_fallback():
    candidates = [
        MODELS_DIR / "cnn_tess_final.keras",
        MODELS_DIR / "cnn_tess_best.keras",
        MODELS_DIR / "cnn_model.h5",
    ]

    for model_path in candidates:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path), model_path

    raise FileNotFoundError("No CNN model found in models directory.")


def main():
    ensure_dirs()
    configure_gpu()

    _, X_mel, y, le, paths, _, _ = load_tess_dataset()
    train_idx, test_idx = get_speaker_split_indices(paths, test_speaker="YAF")

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Speaker split failed. YAF/OAF folders were not found correctly.")

    X_test = X_mel[test_idx]
    y_test = y[test_idx]

    print(f"Test samples: {len(X_test)}")

    model, model_path = load_model_fallback()
    print(f"Loaded model: {model_path}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    with open(os.path.join(RESULTS_DIR, "cnn_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()