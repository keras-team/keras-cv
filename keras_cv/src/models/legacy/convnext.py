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

"""ConvNeXt models for Keras.
References:
- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  (CVPR 2022)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.src.layers.regularization import StochasticDepth
from keras_cv.src.models.legacy import utils

MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "default_size": 224,
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "default_size": 224,
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "default_size": 224,
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (CVPR 2022)

    This function returns a Keras {name} model.
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, `num_classes` must be provided.
        depths: an iterable containing depths for each individual stages.
        projection_dims: An iterable containing output number of channels of
            each individual stages.
        drop_path_rate: stochastic depth probability, if 0.0, then stochastic
            depth won't be used.
        layer_scale_init_value: layer scale coefficient, if 0.0, layer scaling
            won't be used.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g.
            'imagenet/classification')(see available pre-trained weights in
            weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        num_classes: optional int, number of classes to classify images into
            (only to be specified if `include_top` is `True`).
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        name: (Optional) name to pass to the model, defaults to "{name}".

    Returns:
      A `keras.Model` instance.
"""


@keras.utils.register_keras_serializable(package="keras_cv")
class LayerScale(layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def apply_block(
    x,
    projection_dim,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    name=None,
):
    """ConvNeXt block.
    References:
      - https://arxiv.org/abs/2201.03545
      - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Notes:
      In the original ConvNeXt implementation (linked above), the authors use
      `Dense` layers for pointwise convolutions for increased efficiency.
      Following that, this implementation also uses the same.
    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float
        number.
      name: name to path to the keras layer.
    Returns:
      A function representing a ConvNeXtBlock block.
    """  # noqa: E501
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    inputs = x

    x = layers.Conv2D(
        filters=projection_dim,
        kernel_size=7,
        padding="same",
        groups=projection_dim,
        name=name + "_depthwise_conv",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
    x = layers.Dense(4 * projection_dim, name=name + "_pointwise_conv_1")(x)
    x = layers.Activation("gelu", name=name + "_gelu")(x)
    x = layers.Dense(projection_dim, name=name + "_pointwise_conv_2")(x)

    if layer_scale_init_value is not None:
        x = LayerScale(
            layer_scale_init_value,
            projection_dim,
            name=name + "_layer_scale",
        )(x)
    if drop_path_rate:
        layer = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")
        return layer([inputs, x])
    else:
        layer = layers.Activation("linear", name=name + "_identity")
        return inputs + layer(x)


def apply_head(x, num_classes, activation="softmax", name=None):
    """Implementation of classification head of ConvNeXt.
    Args:
      num_classes: number of classes for Dense layer
      activation: activation function for Dense layer
      name: name prefix
    Returns:
      Classification head function.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=name + "_head_layernorm")(
        x
    )
    x = layers.Dense(
        num_classes, activation=activation, name=name + "_head_dense"
    )(x)
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ConvNeXt(keras.Model):
    """Instantiates ConvNeXt architecture given specific configuration.
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, `num_classes` must be provided.
        depths: An iterable containing depths for each individual stages.
        projection_dims: An iterable containing output number of channels of
        each individual stages.
        drop_path_rate: Stochastic depth probability. If 0.0, then stochastic
        depth won't be used.
        layer_scale_init_value: Layer scale coefficient. If 0.0, layer scaling
        won't be used.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g.
            'imagenet/classification')(see available pre-trained weights in
            weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        num_classes: optional int, number of classes to classify images into
            (only to be specified if `include_top` is `True`).
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        name: (Optional) name to pass to the model, defaults to "convnext".
    Returns:
      A `keras.Model` instance.
    Raises:
        ValueError: in case of invalid argument for `weights`, or invalid input
            shape.
        ValueError: if `classifier_activation` is not `softmax`, or `None` when
            using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not specified.
    """

    def __init__(
        self,
        include_rescaling,
        include_top,
        depths,
        projection_dims,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
        name="convnext",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either "
                "`None` or the path to the weights file to be loaded. "
                f"Weights file not found at location: {weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, "
                "you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
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

        # Stem block.
        stem = keras.Sequential(
            [
                layers.Conv2D(
                    projection_dims[0],
                    kernel_size=4,
                    strides=4,
                    name=name + "_stem_conv",
                ),
                layers.LayerNormalization(
                    epsilon=1e-6, name=name + "_stem_layernorm"
                ),
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
        depth_drop_rates = [
            float(x) for x in tf.linspace(0.0, drop_path_rate, sum(depths))
        ]

        # First apply downsampling blocks and then apply ConvNeXt stages.
        cur = 0

        num_convnext_blocks = 4
        for i in range(num_convnext_blocks):
            x = downsample_layers[i](x)
            for j in range(depths[i]):
                x = apply_block(
                    x,
                    projection_dim=projection_dims[i],
                    drop_path_rate=depth_drop_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    name=name + f"_stage_{i}_block_{j}",
                )
            cur += depths[i]

        if include_top:
            x = apply_head(
                x,
                num_classes=num_classes,
                activation=classifier_activation,
                name=name,
            )

        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D()(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        if weights is not None:
            self.load_weights(weights)

        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.depths = depths
        self.projection_dims = projection_dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "depths": self.depths,
            "projection_dims": self.projection_dims,
            "drop_path_rate": self.drop_path_rate,
            "layer_scale_init_value": self.layer_scale_init_value,
            # Remove batch dimension from `input_shape`
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ConvNeXtTiny(
    *,
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
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
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtSmall(
    *,
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
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
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtBase(
    *,
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
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
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtLarge(
    *,
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
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
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
    )


def ConvNeXtXLarge(
    *,
    include_rescaling,
    include_top,
    drop_path_rate,
    layer_scale_init_value,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
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
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
    )


ConvNeXtTiny.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtTiny")
ConvNeXtSmall.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtSmall")
ConvNeXtBase.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtBase")
ConvNeXtLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtLarge")
ConvNeXtXLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtXLarge")
