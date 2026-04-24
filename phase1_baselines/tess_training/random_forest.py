import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import MODELS_DIR, ensure_dirs
from data_utils import get_speaker_split_indices, load_tess_dataset, train_test_split_stratified


def main():
    ensure_dirs()

    X_feat, _, y, le, paths, _, _ = load_tess_dataset()
    train_idx, test_idx = get_speaker_split_indices(paths, test_speaker="YAF")

    if len(train_idx) == 0 or len(test_idx) == 0:
        X_train, X_test, y_train, y_test = train_test_split_stratified(X_feat, y)
    else:
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"Random Forest Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
        },
        MODELS_DIR / "rf_tess.joblib",
    )


if __name__ == "__main__":
    main()