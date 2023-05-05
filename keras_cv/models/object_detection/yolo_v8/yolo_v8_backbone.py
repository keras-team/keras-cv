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
import copy

import tensorflow as tf
from keras import layers
from tensorflow import keras

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_conv_bn,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    apply_csp_block,
)
from keras_cv.utils.python_utils import classproperty


def apply_spatial_pyramid_pooling_fast(
    inputs, pool_size=5, activation="swish", name="spp_fast"
):
    channel_axis = -1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    x = apply_conv_bn(
        inputs,
        hidden_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    pool_1 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool1"
    )(x)
    pool_2 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool2"
    )(pool_1)
    pool_3 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool3"
    )(pool_2)

    out = tf.concat([x, pool_1, pool_2, pool_3], axis=channel_axis)
    out = apply_conv_bn(
        out,
        input_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out


# TODO(ianstenbit) This should probably just be a CSPDarknet
@keras.utils.register_keras_serializable(package="keras_cv.models")
class YOLOV8Backbone(Backbone):
    """Implements the YOLOV8 backbone for object detection.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_channels: A list of ints, the number of channels for each dark
            level in the model.
        stackwise_depth: A list of ints, the depth for each dark level in the
            model.
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        activation: String. The activation functions to use in the backbone.
            Defaults to "swish".
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Returns:
        A `keras.Model` instance.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolov8_xs_backbone_coco"
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_cv.models.YOLOV8Backbone(
        stackwise_channels=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        stackwise_channels,
        stackwise_depth,
        include_rescaling,
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        """ Stem """
        stem_width = stackwise_channels[0]
        x = apply_conv_bn(
            x,
            stem_width // 2,
            kernel_size=3,
            strides=2,
            activation=activation,
            name="stem_1",
        )
        x = apply_conv_bn(
            x,
            stem_width,
            kernel_size=3,
            strides=2,
            activation=activation,
            name="stem_2",
        )

        """ blocks """
        pyramid_level_inputs = {1: x.node.layer.name}
        for stack_id, (channel, depth) in enumerate(
            zip(stackwise_channels, stackwise_depth)
        ):
            stack_name = f"stack{stack_id + 1}"
            if stack_id >= 1:
                x = apply_conv_bn(
                    x,
                    channel,
                    kernel_size=3,
                    strides=2,
                    activation=activation,
                    name=f"{stack_name}_downsample",
                )
            x = apply_csp_block(
                x,
                depth=depth,
                expansion=0.5,
                activation=activation,
                name=f"{stack_name}_c2f",
            )

            if stack_id == len(stackwise_depth) - 1:
                x = apply_spatial_pyramid_pooling_fast(
                    x,
                    pool_size=5,
                    activation=activation,
                    name=f"{stack_name}_spp_fast",
                )
            pyramid_level_inputs[stack_id + 2] = x.node.layer.name

        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_channels = stackwise_channels
        self.stackwise_depth = stackwise_depth
        self.include_rescaling = include_rescaling
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "stackwise_channels": self.stackwise_channels,
                "stackwise_depth": self.stackwise_depth,
                "activation": self.activation,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)
