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

"""DenseNet backbone model.

Reference:
  - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  - [Based on the Original keras.applications DenseNet](https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py)
"""  # noqa: E501

import copy

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.densenet.densenet_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.models.backbones.densenet.densenet_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty

BN_AXIS = 3
BN_EPSILON = 1.001e-5


@keras_cv_export("keras_cv.models.DenseNetBackbone")
class DenseNetBackbone(Backbone):
    """Instantiates the DenseNet architecture.

    Args:
        stackwise_num_repeats: list of ints, number of repeated convolutional
            blocks per dense block.
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of
            `keras.layers.Input()`) to use as image input for the model.
        compression_ratio: float, compression rate at transition layers.
        growth_rate: int, number of filters added by each dense block.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.DenseNetBackbone.from_preset("densenet121_imagenet")
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = DenseNetBackbone(
        stackwise_num_repeats=[6, 12, 24, 16],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stackwise_num_repeats,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        compression_ratio=0.5,
        growth_rate=32,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        x = keras.layers.Conv2D(
            64, 7, strides=2, use_bias=False, padding="same", name="conv1_conv"
        )(x)
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1"
        )(x)

        pyramid_level_inputs = {}
        for stack_index in range(len(stackwise_num_repeats) - 1):
            index = stack_index + 2
            x = apply_dense_block(
                x,
                stackwise_num_repeats[stack_index],
                growth_rate,
                name=f"conv{index}",
            )
            pyramid_level_inputs[f"P{index}"] = utils.get_tensor_input_name(x)
            x = apply_transition_block(
                x, compression_ratio, name=f"pool{index}"
            )

        x = apply_dense_block(
            x,
            stackwise_num_repeats[-1],
            growth_rate,
            name=f"conv{len(stackwise_num_repeats) + 1}",
        )

        pyramid_level_inputs[f"P{len(stackwise_num_repeats) + 1}"] = (
            utils.get_tensor_input_name(x)
        )
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="bn"
        )(x)
        x = keras.layers.Activation("relu", name="relu")(x)

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_num_repeats = stackwise_num_repeats
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.compression_ratio = compression_ratio
        self.growth_rate = growth_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "compression_ratio": self.compression_ratio,
                "growth_rate": self.growth_rate,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return copy.deepcopy(backbone_presets_with_weights)


def apply_dense_block(x, num_repeats, growth_rate, name=None):
    """A dense block.

    Args:
      x: input tensor.
      num_repeats: int, number of repeated convolutional blocks.
      growth_rate: int, number of filters added by each dense block.
      name: string, block label.
    """
    if name is None:
        name = f"dense_block_{keras.backend.get_uid('dense_block')}"

    for i in range(num_repeats):
        x = apply_conv_block(x, growth_rate, name=f"{name}_block_{i}")
    return x


def apply_transition_block(x, compression_ratio, name=None):
    """A transition block.

    Args:
      x: input tensor.
      compression_ratio: float, compression rate at transition layers.
      name: string, block label.
    """
    if name is None:
        name = f"transition_block_{keras.backend.get_uid('transition_block')}"

    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu")(x)
    x = keras.layers.Conv2D(
        int(x.shape[BN_AXIS] * compression_ratio),
        1,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
    return x


def apply_conv_block(x, growth_rate, name=None):
    """A building block for a dense block.

    Args:
      x: input tensor.
      growth_rate: int, number of filters added by each dense block.
      name: string, block label.
    """
    if name is None:
        name = f"conv_block_{keras.backend.get_uid('conv_block')}"

    shortcut = x
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_0_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_0_relu")(x)
    x = keras.layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=f"{name}_1_conv"
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_1_relu")(x)
    x = keras.layers.Conv2D(
        growth_rate,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)
    x = keras.layers.Concatenate(axis=BN_AXIS, name=f"{name}_concat")(
        [shortcut, x]
    )
    return x
