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
"""MaxViT (Multi-Axis Vision Transformer) models for Keras.
Reference:
  - [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697) (ECCV 2022)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers.maxvit_layers import MaxViTBlock
from keras_cv.models import utils

MODEL_CONFIGS = {}

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697) (ECCV 2022)
    This function returns a Keras {name} model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).
    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, classes must be provided.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g. 'imagenet/classification')
            (see available pre-trained weights in weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the model will
                be a 2D tensor.
            - `max` means that global max pooling will be applied.
            - `token_pooling`, default, means that the token at the start of the
                sequences is used instead of regular pooling.
        name: (Optional) name to pass to the model.  Defaults to "{name}".
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
    Returns:
      A `keras.Model` instance.
"""


def MaxViT(
    include_rescaling,
    include_top,
    name="MaxViT",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    num_blocks=None,
    activation=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT architecture.

    Args:
        mlp_dim: the dimensionality of the hidden Dense layer in the transformer
            MLP head
        include_rescaling: whether or not to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
            name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            or the path to the weights file to be loaded.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
            - `token_pooling`, default, means that the token at the start of the
                sequences is used instead of regular pooling.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.
                    mlp_dim:
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        **kwargs: Pass-through keyword arguments to `tf.keras.Model`.

    Returns:
      A `keras.Model` instance.
    """

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
        x = layers.Rescaling(1 / 255.0)(x)

        """
        Input
        Conv 3x3 (s=2)
        Conv 3x3
        4 x (L x MaxViT blocks)
        Pool + FC head
        Output
        """

    x = layers.Conv2D(filters=None, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.Conv2D(filters=None, kernel_size=3, strides=1, activation="relu")(x)

    for _ in range(num_blocks):
        x = MaxViTBlock(
            project_dim=None,
            mlp_dim=None,
            num_heads=None,
            mlp_dropout=None,
            attention_dropout=None,
            activation=None,
        )(x)

    if include_top:
        output = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(classes, activation=classifier_activation)(output)

    else:
        output = layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


def MaxViT_V():
    """Instantiates the ViTTiny16 architecture."""

    return MaxViT()
