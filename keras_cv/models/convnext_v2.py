## Copyright 2022 The KerasCV Authors
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

"""ConvNeXtV2.

References:
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.layers.regularization import GlobalResponseNormalization
from keras_cv.layers.regularization import StochasticDepth
from keras_cv.models import utils

MODEL_CONFIGS = {
    "atto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [40, 80, 160, 320],
    },
    "femto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [48, 96, 192, 384],
    },

    "pico": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [64, 128, 256, 512],
    },
    "nano": {
        "depths": [2, 2, 8, 2],
        "projection_dims": [80, 160, 320, 640],
    },
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
    "huge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [352, 704, 1408, 2816],
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.
    - [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)

    This function returns a Keras {name} model.

    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, classes must be provided.
        depths: an iterable containing depths for each individual stages.
        projection_dims: An iterable containing output number of channels of
            each individual stages.
        drop_path_rate: stochastic depth probability, if 0.0, then stochastic
            depth won't be used.
        weights: one of `None` (random initialization), or a pretrained weight
            file path.
        input_shape: optional shape tuple, defaults to `(None, None, 3)`.
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
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
            Defaults to `"softmax"`.
        name: (Optional) name to pass to the model.  Defaults to "{name}".

    Returns:
      A `keras.Model` instance.
"""


class ConvNeXtV2Block(layers.Layer)
    """ConvNeXV2 block.

    References:
      - https://arxiv.org/abs/2301.00808
      - https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py

    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      name: name to path to the Keras layer.

    Returns:
      A function representing a ConvNeXtBlock block.
    """


    def __init__(self, projection_dim, drop_path_rate=0.0, **kwargs):
        # Depthwise with groups
        self.depthwise_conv = layers.Conv2D(
            filters=projection_dim,
            kernel_size=7,
            padding="same",
            groups=projection_dim
        )

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        # Pointwise/1x1 conv, implemented with Dense layers
        # as per the official implementation
        self.dense = layers.Dense(4 * projection_dim)
        self.gelu = layers.Activation("gelu")
        self.grn = GlobalResponseNormalization()
        self.dense2 = layers.Dense(projection_dim)
        self.drop_path_rate = drop_path_rate
    def call(self, inputs):
        x = inputs

        x = self.depthwise_conv(x)
        x = self.layernorm(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.grn(x)
        x = self.dense2(x)

        if self.drop_path_rate:
            layer = StochasticDepth(self.drop_path_rate)
            return layer([inputs, x])
        else:
            layer = layers.Activation("linear")
            return inputs + layer(x)


def ConvNeXtV2(
    include_rescaling,
    include_top,
    depths,
    projection_dims,
    drop_path_rate=0.0,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    name="convnextv2",
):
    """Instantiates ConvNeXtV2 architecture given specific configuration.

    Args:
      include_rescaling: whether or not to Rescale the inputs. If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
      include_top: Boolean denoting whether to include classification head to
        the model.
      depths: An iterable containing depths for each individual stages.
      projection_dims: An iterable containing output number of channels of
      each individual stages.
      drop_path_rate: Stochastic depth probability. If 0.0, then stochastic
        depth won't be used.
      weights: One of `None` (random initialization), or a pretrained weight
        file path.
      input_shape: optional shape tuple, defaults to `(None, None, 3)`.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`).
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor output
          of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output
          of the last convolutional layer, and thus the output of the model will
          be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
      name: An optional name for the model.

    Returns:
      A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax`, or `None`
          when using a pretrained top layer.
        ValueError: if `include_top` is True but `classes` is not specified.
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

    if include_top and pooling:
        raise ValueError(
            f"`pooling` must be `None` when `include_top=True`."
            f"Received pooling={pooling} and include_top={include_top}. "
        )

    inputs = utils.parse_model_inputs(input_shape, input_tensor)

    x = inputs
    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    stem = keras.Sequential(
        [
            layers.Conv2D(
                projection_dims[0],
                kernel_size=4,
                strides=4,
                name=name + "_stem_conv",
            ),
            layers.LayerNormalization(epsilon=1e-6, name=name + "_stem_layernorm"),
        ],
        name=name + "_stem",
    )

    # Downsampling blocks.
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-6,
                    name=name + "_downsampling_layernorm_" + str(i),
                ),
                layers.Conv2D(
                    projection_dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    name=name + "_downsampling_conv_" + str(i),
                ),
            ],
            name=name + "_downsampling_block_" + str(i),
        )
        downsample_layers.append(downsample_layer)

    # Stochastic depth schedule.
    # This is referred from the original ConvNeXt codebase:
    # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
    depth_drop_rates = [float(x) for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

    # First apply downsampling blocks and then apply ConvNeXt stages.
    cur = 0

    num_convnext_blocks = 4
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtV2Block(
                projection_dim=projection_dims[i],
                drop_path_rate=depth_drop_rates[cur + j],
                name=name + f"_stage_{i}_block_{j}",
            )(x)
        cur += depths[i]

    if include_top:
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_head_layernorm")(x)
        x = layers.Dense(classes, activation=classifier_activation, name=name + "_head_dense")(
            x
        )
        return x

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    model = keras.Model(inputs=inputs, outputs=x, name=name)

    if weights is not None:
        model.load_weights(weights)

    return model


def ConvNeXtTiny(
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    name="convnext_tiny",
):
    return ConvNeXt(
        include_rescaling=include_rescaling,
        include_top=include_top,
        depths=MODEL_CONFIGS["tiny"]["depths"],
        projection_dims=MODEL_CONFIGS["tiny"]["projection_dims"],
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtSmall(
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    name="convnext_small",
):
    return ConvNeXt(
        include_rescaling=include_rescaling,
        include_top=include_top,
        depths=MODEL_CONFIGS["small"]["depths"],
        projection_dims=MODEL_CONFIGS["small"]["projection_dims"],
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtBase(
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    name="convnext_base",
):
    return ConvNeXt(
        include_rescaling=include_rescaling,
        include_top=include_top,
        depths=MODEL_CONFIGS["base"]["depths"],
        projection_dims=MODEL_CONFIGS["base"]["projection_dims"],
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtLarge(
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    name="convnext_large",
):
    return ConvNeXt(
        include_rescaling=include_rescaling,
        include_top=include_top,
        depths=MODEL_CONFIGS["large"]["depths"],
        projection_dims=MODEL_CONFIGS["large"]["projection_dims"],
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtXLarge(
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    name="convnext_xlarge",
):
    return ConvNeXt(
        include_rescaling=include_rescaling,
        include_top=include_top,
        depths=MODEL_CONFIGS["xlarge"]["depths"],
        projection_dims=MODEL_CONFIGS["xlarge"]["projection_dims"],
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        name=name,
    )


ConvNeXtTiny.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtTiny")
ConvNeXtSmall.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtSmall")
ConvNeXtBase.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtBase")
ConvNeXtLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtLarge")
ConvNeXtXLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtXLarge")
