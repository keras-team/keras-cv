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

from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import conv_bn
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    csp_with_2_conv,
)
from keras_cv.utils.python_utils import classproperty


def spatial_pyramid_pooling_fast(
    inputs, pool_size=5, activation="swish", name=None
):
    channel_axis = -1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    nn = conv_bn(
        inputs,
        hidden_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    pool_1 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool1"
    )(nn)
    pool_2 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool2"
    )(pool_1)
    pool_3 = layers.MaxPool2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool3"
    )(pool_2)

    out = tf.concat([nn, pool_1, pool_2, pool_3], axis=channel_axis)
    out = conv_bn(
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
    def __init__(
        self,
        include_rescaling,
        channels=[128, 256, 512, 1024],
        depths=[3, 6, 6, 3],
        input_shape=(512, 512, 3),
        activation="swish",
        **kwargs,
    ):
        inputs = layers.Input(input_shape)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        """ Stem """
        # stem_width = stem_width if stem_width > 0 else channels[0]
        stem_width = channels[0]
        nn = conv_bn(
            x,
            stem_width // 2,
            kernel_size=3,
            strides=2,
            activation=activation,
            name="stem_1",
        )
        nn = conv_bn(
            nn,
            stem_width,
            kernel_size=3,
            strides=2,
            activation=activation,
            name="stem_2",
        )

        """ blocks """
        pyramid_level_inputs = {0: nn.node.layer.name}
        features = {0: nn}
        for stack_id, (channel, depth) in enumerate(zip(channels, depths)):
            stack_name = f"stack{stack_id + 1}"
            if stack_id >= 1:
                nn = conv_bn(
                    nn,
                    channel,
                    kernel_size=3,
                    strides=2,
                    activation=activation,
                    name=f"{stack_name}_downsample",
                )
            nn = csp_with_2_conv(
                nn,
                depth=depth,
                expansion=0.5,
                activation=activation,
                name=f"{stack_name}_c2f",
            )

            if stack_id == len(depths) - 1:
                nn = spatial_pyramid_pooling_fast(
                    nn,
                    pool_size=5,
                    activation=activation,
                    name=f"{stack_name}_spp_fast",
                )
            pyramid_level_inputs[stack_id + 1] = nn.node.layer.name
            features[stack_id + 1] = nn

        super().__init__(inputs=inputs, outputs=features, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""
        return copy.deepcopy(backbone_presets_with_weights)
