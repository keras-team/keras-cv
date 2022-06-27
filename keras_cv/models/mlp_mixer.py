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

"""MLP Mixer models for KerasCV.

Reference:
  - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers


def MLPBlock(mlp_dim, name=None):
    """An MLP block consisting of two linear layers with GELU activation in
    between.

    Args:
      mlp_dim: integer, the number of units to be present in the first layer.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing an MLP block.
    """
    if name is None:
        name = f"mlp_block_{backend.get_uid('mlp_block')}"

    def apply(x):
        y = layers.Dense(mlp_dim, name=f"{name}_dense_1")(x)
        y = layers.Activation("gelu", name=f"{name}_gelu")(y)
        return layers.Dense(x.shape[-1], name=f"{name}_dense_2")(y)

    return apply


def MixerBlock(tokens_mlp_dim, channels_mlp_dim, name=None):
    """A mixer block.

    Args:
      tokens_mlp_dim: integer, number of units to be present in the MLP block
        dealing with tokens.
      channels_mlp_dim: integer, number of units to be present in the MLP block
        dealing with channels.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing an MLP block.
    """
    if name is None:
        name = f"mixer_block_{backend.get_uid('mlp_block')}"

    def apply(x):
        y = layers.LayerNormalization()(x)
        y = layers.Permute((2, 1))(y)

        y = MLPBlock(tokens_mlp_dim, name=f"{name}_token_mixing")(y)
        y = layers.Permute((2, 1))(y)
        x = layers.Add()([x, y])

        y = layers.LayerNormalization()(x)
        y = MLPBlock(channels_mlp_dim, name=f"{name}_channel_mixing")(y)
        return layers.Add()([x, y])

    return apply


def MLPMixer(
    patch_size,
    num_blocks,
    hidden_dim,
    tokens_mlp_dim,
    channels_mlp_dim,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classifier_activation="softmax",
    name=None,
    **kwargs,
):
    """Instantiates the MLP Mixer architecture.

    Reference:
    - [MLP-Mixer: An all-MLP Architecture for Vision (NeurIPS 2021)](
        https://arxiv.org/abs/2105.01601)

    This function returns a Keras MLP Mixer model.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    Args:
      patch_size: size of the patches to be extracted from the inputs.
      num_blocks: number of mixer blocks.
      hidden_dim: dimension to which the patches will be linearly projected.
      tokens_mlp_dim: dimension of the MLP block responsible for tokens.
      channels_mlp_dim: dimension of the MLP block responsible for channels.
      include_rescaling: whether or not to Rescale the inputs.
        If set to True, inputs will be passed through a
        `Rescaling(1/255.0)` layer.
      include_top: whether to include the fully-connected
        layer at the top of the network.  If provided, num_classes must be provided.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      weights: one of `None` (random initialization), or a pretrained
        weight file path.
      input_shape: optional shape tuple, defaults to (None, None, 3).
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      name: (Optional) name to pass to the model.  Defaults to "DenseNet".

    Returns:
      A `keras.Model` instance.
    """
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` or the path to the weights file to be loaded. "
            f"Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, "
            "you should specify `classes`. "
            f"Received: classes={classes}"
        )

    if input_shape[0] % patch_size[0] != 0:
        raise ValueError("Input resolution should be divisible" " by the patch size.")

    inputs = layers.Input(shape=input_shape)

    x = inputs
    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    x = layers.Conv2D(
        filters=hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patchify_and_projection",
    )(x)
    x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

    for i in range(num_blocks):
        x = MixerBlock(tokens_mlp_dim, channels_mlp_dim, name=f"mixer_block_{i}")(x)

    x = layers.LayerNormalization()(x)

    if include_top:
        x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )

    elif pooling == "avg":
        x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling1D(name="max_pool")(x)

    model = keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)
    return model


def MLPMixerB16(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(224, 224, 3),
    pooling=None,
    name="mlp_mixer_b16",
    **kwargs,
):
    return MLPMixer(
        patch_size=(16, 16),
        num_blocks=12,
        hidden_dim=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


def MLPMixerB32(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(224, 224, 3),
    pooling=None,
    name="mlp_mixer_b32",
    **kwargs,
):
    return MLPMixer(
        patch_size=(32, 32),
        num_blocks=12,
        hidden_dim=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


def MLPMixerL16(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(224, 224, 3),
    pooling=None,
    name="mlp_mixer_l16",
    **kwargs,
):
    return MLPMixer(
        patch_size=(16, 16),
        num_blocks=24,
        hidden_dim=1024,
        tokens_mlp_dim=512,
        channels_mlp_dim=4096,
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )
