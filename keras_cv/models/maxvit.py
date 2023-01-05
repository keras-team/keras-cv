# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MaxViT (Multi Axis Vision Transformer) models for Keras.
Reference:
  - [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697) (ECCV 2022)
"""
import tensorflow as tf
from tensorflow.keras import layers

import keras_cv
from keras_cv.layers import MaxViTBlock
from keras_cv.layers import MaxViTStem
from keras_cv.models import utils
from keras_cv.models.weights import parse_weights

MODEL_CONFIGS = {
    "MaxViTTiny": {
        "stem_hsize": [64, 64],
        "head_size": 32,
        "num_blocks": [2, 2, 5, 2],
        "hidden_sizes": [64, 128, 256, 512],
        "window_size": 7,
        "grid_size": 7,
    },
    "MaxViTSmall": {
        "stem_hsize": [64, 64],
        "head_size": 32,
        "num_blocks": [2, 2, 5, 2],
        "hidden_size": [96, 192, 384, 768],
        "window_size": 7,
        "grid_size": 7,
    },
    "MaxViTBase": {
        "stem_hsize": [64, 64],
        "head_size": 32,
        "num_blocks": [2, 6, 14, 2],
        "hidden_size": [96, 192, 384, 768],
        "window_size": 7,
        "grid_size": 7,
    },
    "MaxViTLarge": {
        "stem_hsize": [128, 128],
        "head_size": 32,
        "num_blocks": [2, 6, 14, 2],
        "hidden_size": [128, 256, 512, 1024],
        "window_size": 7,
        "grid_size": 7,
    },
    "MaxViTXLarge": {
        "stem_hsize": [192, 192],
        "head_size": 32,
        "num_blocks": [2, 6, 14, 2],
        "hidden_size": [192, 384, 768, 1536],
        "window_size": 7,
        "grid_size": 7,
    },
}


BASE_DOCSTRING = """..."""


def MaxViT(
    include_rescaling,
    include_top,
    name="MaxViT",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    stem_hsize=None,
    num_blocks=None,
    hidden_sizes=None,
    window_size=None,
    grid_size=None,
    attention_dropout=None,
    classifier_activation="softmax",
    **kwargs,
):
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            "weights file to be loaded. Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, you should specify `classes`. "
            f"Received: classes={classes}"
        )

    if include_top and pooling:
        raise ValueError(
            f"`pooling` must be `None` when `include_top=True`."
            f"Received pooling={pooling} and include_top={include_top}. "
        )

    inputs = utils.parse_model_inputs(input_shape, input_tensor)
    x = inputs

    if include_rescaling:
        x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    stem = MaxViTStem(stem_hsize)
    stem_output = stem(x)
    encoder_input = stem_output

    for i, num_block in enumerate(num_blocks):
        hidden_size = hidden_sizes[i]
        print(hidden_size)
        for j in range(num_block):
            if j == 0:
                pool_stride = 2
            else:
                pool_stride = 1
            encoder_output = MaxViTBlock(
                hidden_size=hidden_size,
                head_size=32,
                window_size=window_size,
                grid_size=grid_size,
                dropout=attention_dropout,
                num_heads=None,
                expansion_rate=4,
                activation="gelu",
                pool_type="avg",
                pool_stride=pool_stride,
                dropatt=None,
                rel_attn_type="2d_multi_head",
                scale_ratio=None,
                survival_prob=None,
                ln_epsilon=1e-5,
                ln_dtype=None,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                bias_initializer=tf.zeros_initializer(),
            )(encoder_input)
            encoder_input = encoder_output

    output = layers.GlobalAveragePooling2D()(encoder_output)
    output = layers.Dense(
        hidden_sizes[-1], activation=tf.keras.layers.Activation(tf.nn.tanh, name="tanh")
    )(output)

    if include_top:
        output = layers.Dense(classes, activation=classifier_activation)(output)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    if weights is not None:
        model.load_weights(weights)

    return model


def MaxViTTiny(
    include_rescaling,
    include_top,
    name="MaxViTTiny",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    stem_hsize=None,
    num_blocks=None,
    hidden_sizes=None,
    window_size=None,
    grid_size=None,
    attention_dropout=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTTiny16 architecture."""

    return MaxViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vittiny16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        stem_hsize=MODEL_CONFIGS["MaxViTTiny"]["stem_hsize"],
        num_blocks=MODEL_CONFIGS["MaxViTTiny"]["num_blocks"],
        hidden_sizes=MODEL_CONFIGS["MaxViTTiny"]["hidden_sizes"],
        window_size=MODEL_CONFIGS["MaxViTTiny"]["window_size"],
        grid_size=MODEL_CONFIGS["MaxViTTiny"]["grid_size"],
        attention_dropout=attention_dropout,
        classifier_activation=classifier_activation,
        **kwargs,
    )
