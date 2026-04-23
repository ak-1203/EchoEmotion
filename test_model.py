from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
SER_MODEL_DIR = PROJECT_ROOT / "ser_model"

if str(SER_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SER_MODEL_DIR))

from ser_model.config import BEST_MODEL_PATH, BATCH_SIZE, EMOTIONS  # noqa: E402
from ser_model.predict import SERPredictor  # noqa: E402

EMOTION_CODE_MAP = {
    "a": "angry",
    "h": "happy",
    "n": "neutral",
    "s": "sad",
}

SENTENCE_TYPE_MAP = {
    "s": "seen",
    "u": "unseen",
}


@dataclass
class FileRecord:
    audio_path: Path
    emotion_code: str
    actor_id: str
    sentence_type_code: str
    sentence_id: str
    true_emotion: str
    sentence_type: str


def parse_filename(audio_path: Path) -> FileRecord:
    parts = audio_path.stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {audio_path.name}")

    emotion_code, actor_id, sentence_type_code, sentence_id = parts

    if emotion_code not in EMOTION_CODE_MAP:
        raise ValueError(f"Unsupported emotion code '{emotion_code}' in {audio_path.name}")
    if sentence_type_code not in SENTENCE_TYPE_MAP:
        raise ValueError(f"Unsupported sentence type '{sentence_type_code}' in {audio_path.name}")

    return FileRecord(
        audio_path=audio_path,
        emotion_code=emotion_code,
        actor_id=actor_id,
        sentence_type_code=sentence_type_code,
        sentence_id=sentence_id,
        true_emotion=EMOTION_CODE_MAP[emotion_code],
        sentence_type=SENTENCE_TYPE_MAP[sentence_type_code],
    )


def discover_audio_files(dataset_root: Path) -> list[FileRecord]:
    audio_paths = sorted(dataset_root.rglob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"No .wav files found under {dataset_root}")
    return [parse_filename(path) for path in audio_paths]


def build_input_tensor(records: list[FileRecord], predictor: SERPredictor) -> torch.Tensor:
    features = []

    for record in records:
        audio, _ = predictor.load_audio(str(record.audio_path))
        mel_spec = predictor.extract_features(audio)
        features.append(mel_spec)

    batch = np.stack(features).astype(np.float32)
    return torch.from_numpy(batch).unsqueeze(1)


def run_inference(
    predictor: SERPredictor,
    batch_tensor: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    probabilities = []
    predictor.model.eval()

    with torch.no_grad():
        for start in range(0, len(batch_tensor), batch_size):
            batch = batch_tensor[start:start + batch_size].to(predictor.device)
            logits = predictor.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)

    return np.concatenate(probabilities, axis=0)


def compute_subset_accuracy(rows: list[dict], sentence_type: str) -> float | None:
    subset = [row for row in rows if row["sentence_type"] == sentence_type]
    if not subset:
        return None
    return float(np.mean([row["correct"] for row in subset]))


def compute_actor_accuracy(rows: list[dict]) -> dict[str, float]:
    per_actor: dict[str, list[bool]] = {}
    for row in rows:
        per_actor.setdefault(row["actor_id"], []).append(bool(row["correct"]))
    return {
        actor_id: float(np.mean(matches))
        for actor_id, matches in sorted(per_actor.items())
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_indices: list[int],
    label_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_bar_plot(values: dict[str, float], title: str, xlabel: str, output_path: Path, color: str) -> None:
    if not values:
        return

    labels = list(values.keys())
    scores = [values[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, scores, color=color)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")

    for label, score in zip(labels, scores):
        ax.text(label, score + 0.02, f"{score:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_predictions_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "audio_path",
        "actor_id",
        "sentence_type",
        "sentence_id",
        "true_emotion",
        "predicted_emotion",
        "confidence",
        "correct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch SER model on EchoEmotion/test_dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "test_dataset",
        help="Root directory containing the Actor* folders.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "test_dataset_evaluation",
        help="Directory for reports, plots, and prediction outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    predictor = SERPredictor(model_path=model_path)
    records = discover_audio_files(dataset_root)
    batch_tensor = build_input_tensor(records, predictor)
    probabilities = run_inference(predictor, batch_tensor, args.batch_size)
    predicted_indices = np.argmax(probabilities, axis=1)

    true_indices = np.array([EMOTIONS.index(record.true_emotion) for record in records], dtype=int)
    predicted_emotions = [EMOTIONS[index] for index in predicted_indices]

    rows = []
    for idx, record in enumerate(records):
        row = asdict(record)
        row["audio_path"] = str(record.audio_path)
        row["predicted_emotion"] = predicted_emotions[idx]
        row["predicted_index"] = int(predicted_indices[idx])
        row["true_index"] = int(true_indices[idx])
        row["confidence"] = float(probabilities[idx][predicted_indices[idx]])
        row["correct"] = bool(predicted_emotions[idx] == record.true_emotion)
        row["all_probabilities"] = {
            emotion: float(probabilities[idx][emotion_idx])
            for emotion_idx, emotion in enumerate(EMOTIONS)
        }
        rows.append(row)

    labels_present = sorted(set(true_indices.tolist()) | set(predicted_indices.tolist()))
    label_names = [EMOTIONS[index] for index in labels_present]

    overall_accuracy = float(accuracy_score(true_indices, predicted_indices))
    seen_accuracy = compute_subset_accuracy(rows, "seen")
    unseen_accuracy = compute_subset_accuracy(rows, "unseen")
    actor_accuracy = compute_actor_accuracy(rows)

    report_text = classification_report(
        true_indices,
        predicted_indices,
        labels=labels_present,
        target_names=label_names,
        zero_division=0,
    )

    save_confusion_matrix(
        true_indices,
        predicted_indices,
        labels_present,
        label_names,
        output_dir / "confusion_matrix.png",
    )

    sentence_type_scores = {
        key: value for key, value in {
            "seen": seen_accuracy,
            "unseen": unseen_accuracy,
        }.items() if value is not None
    }
    save_bar_plot(
        sentence_type_scores,
        title="Seen vs Unseen Accuracy",
        xlabel="Sentence Type",
        output_path=output_dir / "sentence_type_accuracy.png",
        color="#59A14F",
    )
    save_bar_plot(
        actor_accuracy,
        title="Actor-wise Accuracy",
        xlabel="Actor",
        output_path=output_dir / "actor_accuracy.png",
        color="#4C78A8",
    )

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    write_predictions_csv(rows, output_dir / "predictions.csv")

    metrics_payload = {
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "num_files": len(records),
        "overall_accuracy": overall_accuracy,
        "seen_accuracy": seen_accuracy,
        "unseen_accuracy": unseen_accuracy,
        "actor_accuracy": actor_accuracy,
        "labels_in_report": label_names,
        "predictions": rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Files evaluated: {len(records)}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    if seen_accuracy is not None:
        print(f"Seen accuracy: {seen_accuracy:.4f}")
    if unseen_accuracy is not None:
        print(f"Unseen accuracy: {unseen_accuracy:.4f}")
    print(report_text)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
