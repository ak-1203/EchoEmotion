"""
Model Builder Module
Constructs the CNN+BiLSTM+Attention neural network architecture (PyTorch)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import (
    NUM_CLASSES,
    CNN_FILTERS,
    CNN_DROPOUT,
    LSTM_UNITS,
    LSTM_DROPOUT,
    NUM_ATTENTION_HEADS,
    L2_REGULARIZATION,
    DROPOUT_RATE,
    BATCH_NORM,
    LOG_LEVEL,
    LEARNING_RATE,
    OPTIMIZER,
    LABEL_SMOOTHING,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class CNNBlock(nn.Module):
    """Convolutional block with optional BatchNorm and Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if BATCH_NORM:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend(
            [
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SERModel(nn.Module):
    """CNN + BiLSTM + Multi-Head Attention model for SER."""

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = NUM_CLASSES):
        super().__init__()
        _, n_mels, _ = input_shape

        self.cnn1 = CNNBlock(1, CNN_FILTERS[0], dropout_rate=CNN_DROPOUT)
        self.cnn2 = CNNBlock(CNN_FILTERS[0], CNN_FILTERS[1], dropout_rate=CNN_DROPOUT)
        self.cnn3 = CNNBlock(CNN_FILTERS[1], CNN_FILTERS[2], dropout_rate=CNN_DROPOUT)

        # After 3 pooling ops along mel axis: n_mels / 8
        mel_after_pool = max(1, n_mels // 8)
        lstm_input_size = CNN_FILTERS[2] * mel_after_pool

        self.bilstm1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=LSTM_UNITS,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bn_lstm1 = nn.BatchNorm1d(LSTM_UNITS * 2) if BATCH_NORM else nn.Identity()
        self.dropout_lstm1 = nn.Dropout(LSTM_DROPOUT)

        self.bilstm2 = nn.LSTM(
            input_size=LSTM_UNITS * 2,
            hidden_size=LSTM_UNITS // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bn_lstm2 = nn.BatchNorm1d(LSTM_UNITS) if BATCH_NORM else nn.Identity()
        self.dropout_lstm2 = nn.Dropout(DROPOUT_RATE)

        self.attention = nn.MultiheadAttention(embed_dim=LSTM_UNITS, num_heads=NUM_ATTENTION_HEADS, batch_first=True)
        self.bn_attention = nn.BatchNorm1d(LSTM_UNITS) if BATCH_NORM else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(LSTM_UNITS, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256) if BATCH_NORM else nn.Identity(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128) if BATCH_NORM else nn.Identity(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes),
        )

    def _apply_timewise_bn(self, x: torch.Tensor, bn_layer: nn.Module) -> torch.Tensor:
        # x: (batch, time, features) -> BN over feature channels
        x = x.transpose(1, 2)
        x = bn_layer(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, time)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        # Rearrange for LSTM: (B, C, H, W) -> (B, W, C*H)
        x = x.permute(0, 3, 1, 2).contiguous()
        bsz, time_steps, channels, mel_bins = x.shape
        x = x.view(bsz, time_steps, channels * mel_bins)

        x, _ = self.bilstm1(x)
        x = self._apply_timewise_bn(x, self.bn_lstm1)
        x = self.dropout_lstm1(x)

        x, _ = self.bilstm2(x)
        x = self._apply_timewise_bn(x, self.bn_lstm2)
        x = self.dropout_lstm2(x)

        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self._apply_timewise_bn(x, self.bn_attention)

        x = x.mean(dim=1)  # Global average pooling over time
        logits = self.classifier(x)
        return logits


def build_ser_model(input_shape: Tuple[int, int, int]) -> nn.Module:
    logger.info(f"Building SER model with input shape: {input_shape}")
    model = SERModel(input_shape=input_shape)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built successfully. Total parameters: {total_params:,}")
    return model


def build_training_components(
    model: nn.Module,
    learning_rate: float = LEARNING_RATE,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = LABEL_SMOOTHING,
):
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    optimizer_name = OPTIMIZER.lower()
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=L2_REGULARIZATION)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_REGULARIZATION)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_REGULARIZATION)

    return criterion, optimizer


if __name__ == '__main__':
    dummy_input_shape = (1, 64, 93)
    model = build_ser_model(dummy_input_shape)
    criterion, optimizer = build_training_components(model)

    dummy_input = torch.randn(1, *dummy_input_shape)
    output = model(dummy_input)

    print("\nModel test passed!")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Expected shape: (1, {NUM_CLASSES})")
