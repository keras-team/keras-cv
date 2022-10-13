from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mlp_block(dropout_rate: float, hidden_units: List[int], name: str = "mlp"):
    """FFN for a Transformer block."""
    ffn = keras.Sequential(name=name)
    for (idx, units) in enumerate(hidden_units):
        ffn.add(
            layers.Dense(
                units,
                activation=tf.nn.gelu if idx == 0 else None,
                bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
            )
        )
        ffn.add(layers.Dropout(dropout_rate))
    return ffn