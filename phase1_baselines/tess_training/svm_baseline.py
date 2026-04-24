import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from config import ID_TO_CLASS, MODELS_DIR, RESULTS_DIR, ensure_dirs
from data_utils import get_speaker_split_indices, load_tess_dataset, prepare_ml_data, train_test_split_stratified


def main():
    ensure_dirs()

    X_feat, _, y, le, paths, _, _ = load_tess_dataset()
    train_idx, test_idx = get_speaker_split_indices(paths, test_speaker="YAF")

    if len(train_idx) == 0 or len(test_idx) == 0:
        X_train, X_test, y_train, y_test = train_test_split_stratified(X_feat, y)
        scaler = None
    else:
        X_train, X_test, y_train, y_test, scaler = prepare_ml_data(X_feat, y, train_idx, test_idx)

    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"SVM Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "label_encoder": le,
        },
        MODELS_DIR / "svm_tess.joblib",
    )


if __name__ == "__main__":
    main()