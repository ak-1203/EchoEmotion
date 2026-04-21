"""
Model Builder Module
Constructs the CNN+BiLSTM+Attention neural network architecture
"""

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
from typing import Tuple
import numpy as np

from config import (
    NUM_CLASSES, CNN_FILTERS, CNN_KERNEL_SIZE, CNN_POOL_SIZE, CNN_DROPOUT,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_RECURRENT_DROPOUT,
    NUM_ATTENTION_HEADS, ATTENTION_HEAD_DIM, L2_REGULARIZATION,
    DROPOUT_RATE, BATCH_NORM, N_MELS, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class MultiHeadAttention(layers.Layer):
    """Custom Multi-Head Attention Layer"""
    
    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        """
        Initialize attention layer
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)
        self.fc_out = layers.Dense(embed_dim)
    
    def call(self, value, key, query, mask=None):
        """Forward pass"""
        batch_size = tf.shape(query)[0]
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, -1, self.num_heads, self.head_dim))
        Q = tf.transpose(Q, (0, 2, 1, 3))
        
        K = tf.reshape(K, (batch_size, -1, self.num_heads, self.head_dim))
        K = tf.transpose(K, (0, 2, 1, 3))
        
        V = tf.reshape(V, (batch_size, -1, self.num_heads, self.head_dim))
        V = tf.transpose(V, (0, 2, 1, 3))
        
        # Attention scores
        scores = tf.matmul(Q, K, transpose_b=True) * self.scale
        
        if mask is not None:
            scores += mask * -1e9
        
        # Softmax and apply to values
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attention_weights, V)
        
        # Concatenate heads
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, -1, self.embed_dim))
        
        # Final linear layer
        output = self.fc_out(context)
        
        return output, attention_weights


class CNNBlock(layers.Layer):
    """Convolutional Block with BatchNorm and Dropout"""
    
    def __init__(self, filters: int, kernel_size: Tuple[int, int], 
                 pool_size: Tuple[int, int], dropout_rate: float = 0.3, **kwargs):
        """
        Initialize CNN block
        
        Args:
            filters: Number of convolutional filters
            kernel_size: Size of convolution kernel
            pool_size: Size of max pooling kernel
            dropout_rate: Dropout rate
        """
        super(CNNBlock, self).__init__(**kwargs)
        
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.L2(L2_REGULARIZATION)
        )
        
        if BATCH_NORM:
            self.bn = layers.BatchNormalization()
        else:
            self.bn = None
        
        self.pool = layers.MaxPooling2D(pool_size=pool_size)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        """Forward pass"""
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x, training=training)
        
        x = self.pool(x)
        x = self.dropout(x, training=training)
        
        return x


def build_ser_model(input_shape: Tuple[int, int, int]) -> Model:
    """
    Build CNN+BiLSTM+Attention model for Speech Emotion Recognition
    
    Args:
        input_shape: Shape of input spectrograms (n_mels, time_steps, 1)
        
    Returns:
        Compiled Keras model
    """
    logger.info(f"Building SER model with input shape: {input_shape}")
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # ========== CNN BLOCKS ==========
    logger.info("Adding CNN blocks...")
    
    for i, filters in enumerate(CNN_FILTERS):
        logger.info(f"  CNN Block {i+1}: {filters} filters")
        cnn_block = CNNBlock(
            filters=filters,
            kernel_size=CNN_KERNEL_SIZE,
            pool_size=CNN_POOL_SIZE,
            dropout_rate=CNN_DROPOUT,
            name=f"cnn_block_{i+1}"
        )
        x = cnn_block(x, training=True)
    
    # Reshape for LSTM: (batch, time, features)
    # From (batch, height, width, channels) to (batch, width, height*channels)
    logger.info("Reshaping for LSTM...")
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)
    
    # ========== BiLSTM LAYERS ==========
    logger.info("Adding BiLSTM layers...")
    
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            kernel_regularizer=regularizers.L2(L2_REGULARIZATION)
        ),
        name="bilstm_1"
    )(x)
    
    if BATCH_NORM:
        x = layers.BatchNormalization(name="bn_bilstm_1")(x)
    
    x = layers.Dropout(DROPOUT_RATE, name="dropout_bilstm_1")(x)
    
    # Second BiLSTM layer
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS // 2,
            return_sequences=True,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            kernel_regularizer=regularizers.L2(L2_REGULARIZATION)
        ),
        name="bilstm_2"
    )(x)
    
    if BATCH_NORM:
        x = layers.BatchNormalization(name="bn_bilstm_2")(x)
    
    x = layers.Dropout(DROPOUT_RATE, name="dropout_bilstm_2")(x)
    
    # ========== ATTENTION MECHANISM ==========
    logger.info("Adding attention mechanism...")
    
    attention_output, attention_weights = MultiHeadAttention(
        embed_dim=LSTM_UNITS,
        num_heads=NUM_ATTENTION_HEADS,
        name="multi_head_attention"
    )(x, x, x)
    
    # Residual connection
    x = layers.Add(name="attention_residual")([x, attention_output])
    
    if BATCH_NORM:
        x = layers.BatchNormalization(name="bn_attention")(x)
    
    # Global average pooling over time dimension
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    
    # ========== DENSE LAYERS ==========
    logger.info("Adding dense layers...")
    
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.L2(L2_REGULARIZATION),
        name="dense_1"
    )(x)
    
    if BATCH_NORM:
        x = layers.BatchNormalization(name="bn_dense_1")(x)
    
    x = layers.Dropout(DROPOUT_RATE, name="dropout_dense_1")(x)
    
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.L2(L2_REGULARIZATION),
        name="dense_2"
    )(x)
    
    if BATCH_NORM:
        x = layers.BatchNormalization(name="bn_dense_2")(x)
    
    x = layers.Dropout(DROPOUT_RATE, name="dropout_dense_2")(x)
    
    # Output layer
    outputs = layers.Dense(
        NUM_CLASSES,
        activation='softmax',
        name="emotion_output"
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Attention_SER")
    
    logger.info(f"Model built successfully!")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    return model


def compile_model(model: Model, learning_rate: float = 0.0005) -> Model:
    """
    Compile the model with optimizer and loss function
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    logger.info(f"Compiling model with learning rate: {learning_rate}")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model compiled successfully!")
    
    return model


if __name__ == '__main__':
    # Test model building
    from config import N_MELS, LEARNING_RATE
    
    # Create dummy input shape (n_mels, time_steps, channels)
    dummy_input_shape = (N_MELS, 94, 1)  # 94 time steps for 3-second audio
    
    # Build model
    model = build_ser_model(dummy_input_shape)
    
    # Compile model
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    # Print model summary
    model.summary()
    
    # Test forward pass
    dummy_input = np.random.randn(1, *dummy_input_shape).astype(np.float32)
    output = model(dummy_input)
    
    print(f"\nModel test passed!")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (1, {NUM_CLASSES})")
