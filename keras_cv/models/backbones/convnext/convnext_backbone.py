# Copyright 2023 The KerasCV Authors
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
import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.layers.regularization import StochasticDepth
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.convnext.convnext_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.legacy import utils
from keras_cv.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_cv")
class LayerScale(layers.Layer):
    """Layer scale module.
    References:
        - https://arxiv.org/abs/2103.17239
    Args:
        init_values: float, initial value for layer scale. Should be within
            [0, 1].
        projection_dim: int, projection dimensionality.
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
        x: input tensor
        projection_dim: int, number of filters for convolution layers. In the
            ConvNeXt paper, this is referred to as projection dimension.
        drop_path_rate: float, probability of dropping paths. Should be within
            [0, 1].
        layer_scale_init_value: float, layer scale value. Should be a small
            float number.
        name: string, name to path to the keras layer.
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
        x: input tensor
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
class ConvNeXtBackbone(Backbone):
    """Instantiates ConvNeXt architecture given specific configuration.
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        stackwise_depths: An iterable containing depths for each individual
            stages.
        stackwise_projection_dims: An iterable containing output number of
            channels of each individual stages.
        drop_path_rate: Stochastic depth probability. If 0.0, then stochastic
            depth won't be used.
        layer_scale_init_value: Layer scale coefficient. If 0.0, layer scaling
            won't be used.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = ConvNeXtBackbone(
        stackwise_depths=[3, 3, 9, 3],
        stackwise_projection_dims=[96, 192, 384, 768],
        default_size=224,
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        *,
        stackwise_depths,
        stackwise_projection_dims,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        # Stem block.
        stem = keras.Sequential(
            [
                layers.Conv2D(
                    stackwise_projection_dims[0],
                    kernel_size=4,
                    strides=4,
                    name="stem_conv",
                ),
                layers.LayerNormalization(epsilon=1e-6, name="stem_layernorm"),
            ],
            name="stem",
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
                        name="downsampling_layernorm_" + str(i),
                    ),
                    layers.Conv2D(
                        stackwise_projection_dims[i + 1],
                        kernel_size=2,
                        strides=2,
                        name="downsampling_conv_" + str(i),
                    ),
                ],
                name="downsampling_block_" + str(i),
            )
            downsample_layers.append(downsample_layer)

        # Stochastic depth schedule.
        # This is referred from the original ConvNeXt codebase:
        # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
        depth_drop_rates = [
            float(x)
            for x in tf.linspace(0.0, drop_path_rate, sum(stackwise_depths))
        ]

        # First apply downsampling blocks and then apply ConvNeXt stages.
        cur = 0

        num_convnext_blocks = 4
        pyramid_level_inputs = {}
        for i in range(num_convnext_blocks):
            x = downsample_layers[i](x)
            for j in range(stackwise_depths[i]):
                x = apply_block(
                    x,
                    projection_dim=stackwise_projection_dims[i],
                    drop_path_rate=depth_drop_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    name=f"stage_{i}_block_{j}",
                )
            cur += stackwise_depths[i]
            pyramid_level_inputs[i + 2] = x.node.layer.name

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.pyramid_level_inputs = pyramid_level_inputs
        self.include_rescaling = include_rescaling
        self.stackwise_depths = stackwise_depths
        self.projection_dims = stackwise_projection_dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "stackwise_depths": self.stackwise_depths,
                "stackwise_projection_dims": self.stackwise_projection_dims,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)


ALIAS_DOCSTRING = """{name} backbone model.

    Reference:
        - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
        (CVPR 2022)

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        stackwise_depths: an iterable containing depths for each individual
            stages.
        stackwise_projection_dims: An iterable containing output number of
            channels of each individual stages.
        drop_path_rate: stochastic depth probability, if 0.0, then stochastic
            depth won't be used.
        layer_scale_init_value: layer scale coefficient, if 0.0, layer scaling
            won't be used.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = {name}Backbone()
    output = model(input_data)
    ```
"""


class ConvNeXtTinyBackbone(ConvNeXtBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvNeXtBackbone.from_preset("convnext_tiny", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvNeXtSmallBackbone(ConvNeXtBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvNeXtBackbone.from_preset("convnext_small", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvNeXtBaseBackbone(ConvNeXtBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvNeXtBackbone.from_preset("convnext_base", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvNeXtLargeBackbone(ConvNeXtBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvNeXtBackbone.from_preset("convnext_large", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvNeXtXLargeBackbone(ConvNeXtBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvNeXtBackbone.from_preset("convnext_xlarge", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


ConvNeXtTinyBackbone.__doc__ = ALIAS_DOCSTRING.format(name="ConvNeXtTiny")
ConvNeXtSmallBackbone.__doc__ = ALIAS_DOCSTRING.format(name="ConvNeXtSmall")
ConvNeXtBaseBackbone.__doc__ = ALIAS_DOCSTRING.format(name="ConvNeXtBase")
ConvNeXtLargeBackbone.__doc__ = ALIAS_DOCSTRING.format(name="ConvNeXtLarge")
ConvNeXtXLargeBackbone.__doc__ = ALIAS_DOCSTRING.format(name="ConvNeXtXLarge")
