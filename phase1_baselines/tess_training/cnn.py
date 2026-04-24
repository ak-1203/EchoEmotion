"""
cnn_train_fixed_v2.py
---------------------
v1 replaced the speaker split with a random split → 100% test accuracy
(data leakage: same utterances from both speakers in train AND test).

v2 restores the speaker split (OAF train / YAF test) which is the only
honest evaluation for a 2-speaker dataset, and keeps all regularisation
fixes so the model actually generalises across speakers.

Key insight: the speaker split IS the right eval. The original model just
had zero regularisation. With SpecAugment + mixup + label smoothing +
higher dropout it should now generalise to the unseen speaker.

Expected result: 80-90% on YAF test set (happy should be 70%+).
"""

import json

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import MODELS_DIR, RESULTS_DIR, ensure_dirs
from data_utils import build_deep_dataset, get_speaker_split_indices, load_tess_dataset
from gpu_utils import configure_gpu


# ──────────────────────────────────────────────────────────────
# SpecAugment
# ──────────────────────────────────────────────────────────────
class SpecAugment(layers.Layer):
    def __init__(
        self,
        freq_mask_param=15,
        time_mask_param=20,
        n_freq_masks=2,
        n_time_masks=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks    = n_freq_masks
        self.n_time_masks    = n_time_masks

    def call(self, x, training=None):
        if not training:
            return x

        shape      = tf.shape(x)
        batch_size = shape[0]
        freq_bins  = shape[1]
        time_steps = shape[2]

        for _ in range(self.n_freq_masks):
            f  = tf.random.uniform((batch_size,), 0, self.freq_mask_param + 1, dtype=tf.int32)
            f0 = tf.random.uniform(
                (batch_size,), 0,
                tf.maximum(1, freq_bins - self.freq_mask_param),
                dtype=tf.int32,
            )
            idx  = tf.range(freq_bins)[tf.newaxis, :, tf.newaxis, tf.newaxis]
            mask = tf.logical_or(idx < f0[:, tf.newaxis, tf.newaxis, tf.newaxis],
                                 idx >= (f0 + f)[:, tf.newaxis, tf.newaxis, tf.newaxis])
            x    = x * tf.cast(mask, x.dtype)

        for _ in range(self.n_time_masks):
            t  = tf.random.uniform((batch_size,), 0, self.time_mask_param + 1, dtype=tf.int32)
            t0 = tf.random.uniform(
                (batch_size,), 0,
                tf.maximum(1, time_steps - self.time_mask_param),
                dtype=tf.int32,
            )
            idx  = tf.range(time_steps)[tf.newaxis, tf.newaxis, :, tf.newaxis]
            mask = tf.logical_or(idx < t0[:, tf.newaxis, tf.newaxis, tf.newaxis],
                                 idx >= (t0 + t)[:, tf.newaxis, tf.newaxis, tf.newaxis])
            x    = x * tf.cast(mask, x.dtype)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            freq_mask_param=self.freq_mask_param,
            time_mask_param=self.time_mask_param,
            n_freq_masks=self.n_freq_masks,
            n_time_masks=self.n_time_masks,
        )
        return cfg


# ──────────────────────────────────────────────────────────────
# Mixup
# ──────────────────────────────────────────────────────────────
def mixup_batch(X, y, num_classes, alpha=0.3, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n   = len(X)
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)
    idx = rng.permutation(n)
    X_m  = lam[:, None, None, None] * X + (1 - lam[:, None, None, None]) * X[idx]
    y_oh = np.eye(num_classes, dtype=np.float32)[y]
    y_m  = lam[:, None] * y_oh + (1 - lam[:, None]) * y_oh[idx]
    return X_m.astype(np.float32), y_m.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────
def build_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = SpecAugment(
        freq_mask_param=15, time_mask_param=20,
        n_freq_masks=2, n_time_masks=2, name="spec_augment",
    )(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.60)(x)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    configure_gpu()

    samples, le, max_mel_len, _ = load_tess_dataset()
    num_classes = len(le.classes_)
    rng         = np.random.default_rng(42)

    paths = [s["path"] for s in samples]

    # ── Restore speaker split (OAF=train+val, YAF=test) ──────────────────
    # This is the honest evaluation: train on one speaker, test on the other.
    # The random split in v1 caused leakage (same utterances, both speakers).
    train_idx, test_idx = get_speaker_split_indices(paths, test_speaker="YAF")

    print(f"OAF (train+val): {len(train_idx)} | YAF (test): {len(test_idx)}")

    # Sanity check — confirm test set is ONLY YAF
    test_speakers = set(s["speaker"] for s in [samples[i] for i in test_idx])
    train_speakers = set(s["speaker"] for s in [samples[i] for i in train_idx])
    print(f"Train speakers : {train_speakers}")
    print(f"Test speakers  : {test_speakers}")
    assert test_speakers == {"YAF"}, "Test set contaminated with non-YAF speaker!"
    assert "YAF" not in train_speakers, "YAF leaked into training set!"

    # Split OAF into train / val
    train_labels = [samples[i]["y"] for i in train_idx]
    oaf_train_idx, oaf_val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=42,
        stratify=train_labels,
    )

    print(f"OAF train: {len(oaf_train_idx)} | OAF val: {len(oaf_val_idx)} | YAF test: {len(test_idx)}")

    # ── Build datasets ────────────────────────────────────────────────────
    X_train, _, y_train = build_deep_dataset(
        samples, oaf_train_idx,
        max_mel_len=max_mel_len, max_seq_len=1,
        augment=True,
        augment_factor=2,           # more augmentation since we only have 1 speaker
        augment_happy_factor=3,
    )
    X_val, _, y_val = build_deep_dataset(
        samples, oaf_val_idx,
        max_mel_len=max_mel_len, max_seq_len=1, augment=False,
    )
    X_test, _, y_test = build_deep_dataset(
        samples, test_idx,
        max_mel_len=max_mel_len, max_seq_len=1, augment=False,
    )

    y_train = y_train.astype(np.int32)
    y_val   = y_val.astype(np.int32)
    y_test  = y_test.astype(np.int32)

    print(f"\nAfter augmentation — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("Train class distribution:")
    for cls_id, cls_name in enumerate(le.classes_):
        print(f"  {cls_name:<15}: {np.sum(y_train == cls_id)}")

    # ── Mixup ─────────────────────────────────────────────────────────────
    X_train_m, y_train_soft = mixup_batch(
        X_train, y_train, num_classes, alpha=0.3, rng=rng
    )

    # ── Class weights ─────────────────────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights)}
    print("\nClass weights:", class_weight_dict)

    # ── Compile ───────────────────────────────────────────────────────────
    model = build_cnn(X_train.shape[1:], num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    y_val_onehot = np.eye(num_classes, dtype=np.float32)[y_val]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
        ModelCheckpoint(
            filepath=MODELS_DIR / "cnn_tess_best_v2.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train_m, y_train_soft,
        validation_data=(X_val, y_val_onehot),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # ── Evaluate on YAF (unseen speaker) ─────────────────────────────────
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\nCNN v2 Accuracy on YAF (unseen speaker): {acc:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

    print("\nPer-class accuracy:")
    for i, cls_name in enumerate(le.classes_):
        correct = cm[i, i]
        total   = cm[i].sum()
        print(f"  {cls_name:<15}: {correct:3d}/{total:3d}  ({correct/total if total else 0:.1%})")

    # ── Save ──────────────────────────────────────────────────────────────
    model.save(MODELS_DIR / "cnn_tess_final_v2.keras")

    with open(RESULTS_DIR / "cnn_v2_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )
    with open(RESULTS_DIR / "cnn_v2_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy (YAF unseen speaker): {acc:.4f}\n\n")
        f.write(report)

    print("\nDone. Model saved to", MODELS_DIR / "cnn_tess_final_v2.keras")


if __name__ == "__main__":
    main()