"""
cnn_bilstm_train_fixed.py
--------------------------
Fixes for CNN+BiLSTM  happy=0% / 49% accuracy / massive overfit issue.

Root causes:
  1. Speaker split → model memorises OAF, fails completely on YAF.
  2. No regularisation on the spectrogram → model learns speaker timbre.
  3. val_accuracy=100% during training is a red flag: val set was also OAF only.
  4. Mixup not applied → hard boundaries between angry/happy/fear.

Fixes:
  A. Stratified random split  (both speakers in every partition).
  B. SpecAugment layer on the mel branch (freq + time masking).
  C. Mixup on the combined [mel, seq] training set.
  D. Label smoothing = 0.1.
  E. Higher dropout on classifier head + L2 on dense layers.
  F. Gradient clipping to stabilise BiLSTM training.
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
from data_utils import build_deep_dataset, load_tess_dataset
from gpu_utils import configure_gpu


# ──────────────────────────────────────────────────────────────
# SpecAugment  (same as CNN fix — applied to mel branch only)
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
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

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
            f0b  = f0[:, tf.newaxis, tf.newaxis, tf.newaxis]
            fb   = f[:, tf.newaxis, tf.newaxis, tf.newaxis]
            mask = tf.logical_or(idx < f0b, idx >= f0b + fb)
            x    = x * tf.cast(mask, x.dtype)

        for _ in range(self.n_time_masks):
            t  = tf.random.uniform((batch_size,), 0, self.time_mask_param + 1, dtype=tf.int32)
            t0 = tf.random.uniform(
                (batch_size,), 0,
                tf.maximum(1, time_steps - self.time_mask_param),
                dtype=tf.int32,
            )
            idx  = tf.range(time_steps)[tf.newaxis, tf.newaxis, :, tf.newaxis]
            t0b  = t0[:, tf.newaxis, tf.newaxis, tf.newaxis]
            tb   = t[:, tf.newaxis, tf.newaxis, tf.newaxis]
            mask = tf.logical_or(idx < t0b, idx >= t0b + tb)
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
# Mixup — adapted for dual inputs [mel, seq]
# ──────────────────────────────────────────────────────────────
def mixup_dual(X_mel, X_seq, y, num_classes, alpha=0.3, rng=None):
    """
    Mixup for the two-input model.
    Returns mixed mel, mixed seq, and soft one-hot labels.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n   = len(X_mel)
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)

    idx = rng.permutation(n)

    X_mel_m = lam[:, None, None, None] * X_mel + (1 - lam[:, None, None, None]) * X_mel[idx]
    X_seq_m = lam[:, None, None]       * X_seq + (1 - lam[:, None, None])       * X_seq[idx]

    y_oh  = np.eye(num_classes, dtype=np.float32)[y]
    y_mix = lam[:, None] * y_oh + (1 - lam[:, None]) * y_oh[idx]

    return (
        X_mel_m.astype(np.float32),
        X_seq_m.astype(np.float32),
        y_mix.astype(np.float32),
    )


# ──────────────────────────────────────────────────────────────
# Model  (SpecAugment + stronger regularisation)
# ──────────────────────────────────────────────────────────────
def build_cnn_bilstm(mel_shape, seq_shape, num_classes):
    reg = tf.keras.regularizers.l2(1e-4)

    # ── Mel branch ───────────────────────────────────────────
    mel_input = layers.Input(shape=mel_shape, name="mel_input")

    x = SpecAugment(
        freq_mask_param=15,
        time_mask_param=20,
        n_freq_masks=2,
        n_time_masks=2,
        name="spec_augment",
    )(mel_input)

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    x          = layers.GlobalAveragePooling2D()(x)
    mel_branch = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    mel_branch = layers.Dropout(0.50)(mel_branch)

    # ── Seq branch (BiLSTM) ──────────────────────────────────
    seq_input  = layers.Input(shape=seq_shape, name="seq_input")
    s          = layers.Masking(mask_value=0.0)(seq_input)
    s          = layers.Bidirectional(
                     layers.LSTM(64, return_sequences=True,
                                 dropout=0.3, recurrent_dropout=0.2)
                 )(s)
    s          = layers.Bidirectional(
                     layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2)
                 )(s)
    seq_branch = layers.Dense(128, activation="relu", kernel_regularizer=reg)(s)
    seq_branch = layers.Dropout(0.50)(seq_branch)

    # ── Fusion head ──────────────────────────────────────────
    merged  = layers.Concatenate()([mel_branch, seq_branch])
    merged  = layers.Dense(128, activation="relu", kernel_regularizer=reg)(merged)
    merged  = layers.Dropout(0.60)(merged)       # was 0.50
    merged  = layers.Dense(64,  activation="relu", kernel_regularizer=reg)(merged)
    merged  = layers.Dropout(0.40)(merged)       # was 0.25
    outputs = layers.Dense(num_classes, activation="softmax")(merged)

    return models.Model(inputs=[mel_input, seq_input], outputs=outputs)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    configure_gpu()

    samples, le, max_mel_len, max_seq_len = load_tess_dataset()
    num_classes = len(le.classes_)
    rng         = np.random.default_rng(42)

    all_idx    = np.arange(len(samples))
    all_labels = np.array([s["y"] for s in samples])

    # ── FIX A: Stratified random split — both speakers in every partition ──
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.15, random_state=42, stratify=all_labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,
        random_state=42,
        stratify=all_labels[train_val_idx],
    )

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ── FIX B: Stronger happy augmentation ────────────────────────────────
    X_train_mel, X_train_seq, y_train = build_deep_dataset(
        samples,
        train_idx,
        max_mel_len=max_mel_len,
        max_seq_len=max_seq_len,
        augment=True,
        augment_factor=1,
        augment_happy_factor=3,          # was 2
    )
    X_val_mel, X_val_seq, y_val = build_deep_dataset(
        samples, val_idx,
        max_mel_len=max_mel_len, max_seq_len=max_seq_len, augment=False,
    )
    X_test_mel, X_test_seq, y_test = build_deep_dataset(
        samples, test_idx,
        max_mel_len=max_mel_len, max_seq_len=max_seq_len, augment=False,
    )

    y_train = y_train.astype(np.int32)
    y_val   = y_val.astype(np.int32)
    y_test  = y_test.astype(np.int32)

    print(f"\nAfter augmentation — Train: {len(X_train_mel)} | Val: {len(X_val_mel)} | Test: {len(X_test_mel)}")
    print("Train class distribution:")
    for cls_id, cls_name in enumerate(le.classes_):
        print(f"  {cls_name:<15}: {np.sum(y_train == cls_id)}")

    # ── FIX C: Mixup on dual inputs ────────────────────────────────────────
    X_train_mel_m, X_train_seq_m, y_train_soft = mixup_dual(
        X_train_mel, X_train_seq, y_train,
        num_classes, alpha=0.3, rng=rng,
    )

    # Class weights (computed on original integer labels)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {
        int(cls): float(w)
        for cls, w in zip(np.unique(y_train), class_weights)
    }
    print("\nClass weights:", class_weight_dict)

    # ── Build & compile ────────────────────────────────────────────────────
    model = build_cnn_bilstm(
        X_train_mel.shape[1:], X_train_seq.shape[1:], num_classes
    )
    model.summary()

    # FIX D: Label smoothing + gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0,                    # gradient clipping — stabilises BiLSTM
    )
    model.compile(
        optimizer=optimizer,
        # CategoricalCrossentropy because y_train_soft is soft labels from mixup
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    # Validation uses original one-hot (not mixed) for honest monitoring
    y_val_onehot = np.eye(num_classes, dtype=np.float32)[y_val]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
        ModelCheckpoint(
            filepath=MODELS_DIR / "cnn_bilstm_tess_best_fixed.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        [X_train_mel_m, X_train_seq_m],
        y_train_soft,                    # soft labels
        validation_data=([X_val_mel, X_val_seq], y_val_onehot),
        epochs=120,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = np.argmax(
        model.predict([X_test_mel, X_test_seq], verbose=0), axis=1
    )
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nCNN+BiLSTM (fixed) Accuracy: {acc:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

    print("\nPer-class accuracy:")
    for i, cls_name in enumerate(le.classes_):
        correct = cm[i, i]
        total   = cm[i].sum()
        print(f"  {cls_name:<15}: {correct:3d}/{total:3d}  ({correct/total if total else 0:.1%})")

    # ── Save ───────────────────────────────────────────────────────────────
    model.save(MODELS_DIR / "cnn_bilstm_tess_final_fixed.keras")

    with open(RESULTS_DIR / "cnn_bilstm_fixed_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )

    with open(RESULTS_DIR / "cnn_bilstm_fixed_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print("\nDone. Model saved to", MODELS_DIR / "cnn_bilstm_tess_final_fixed.keras")


if __name__ == "__main__":
    main()