from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Generic MLP for transformer heads.
"""
def mlp_head(x, dropout_rate: float, hidden_units: List[int]):
    for (idx, units) in enumerate(hidden_units):
        x = layers.Dense(units, activation=tf.nn.gelu if idx == 0 else None)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
