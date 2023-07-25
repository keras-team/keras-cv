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
import copy

from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.object_detection_3d.center_pillar_backbone_presets import (
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_cv.models")
class CenterPillarBackbone(Backbone):
    """A UNet backbone for CenterPillar models.

    All up and down blocks scale by a factor of two. Skip connections are
    included.

    All function parameters require curried functions as inputs which return a
    function that acts on tensors as inputs.

    Reference: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

    Args:
        stackwise_down_blocks: a list of integers representing the number of
            sub-blocks in each downsampling block.
        stackwise_down_filters: a list of integers representing the number of
            filters in each downsampling block.
        stackwise_up_filters: a list of integers representing the number of
            filters in each upsampling block.
        input_shape: the rank 3 shape of the input to the UNet.
    """  # noqa: E501

    def __init__(
        self,
        stackwise_down_blocks,
        stackwise_down_filters,
        stackwise_up_filters,
        input_shape=(None, None, 128),
        **kwargs
    ):
        input = layers.Input(shape=input_shape)
        x = input

        x = keras.layers.Conv2D(
            128,
            1,
            1,
            padding="same",
            kernel_initializer=keras.initializers.VarianceScaling(),
            kernel_regularizer=keras.regularizers.L2(l2=1e-4),
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = Block(128, downsample=False)(x)

        skip_connections = []
        # Filters refers to the number of convolutional filters in each block,
        # while num_blocks refers to the number of sub-blocks within a block
        # (Note that only the first sub-block will perform downsampling)
        for filters, num_blocks in zip(
            stackwise_down_filters, stackwise_down_blocks
        ):
            skip_connections.append(x)
            x = DownSampleBlock(filters, num_blocks)(x)

        for filters in stackwise_up_filters:
            x = UpSampleBlock(filters)(x, skip_connections.pop())

        output = x

        super().__init__(
            inputs=input,
            outputs=output,
            **kwargs,
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)


def Block(filters, downsample):
    """A default block which serves as an example of the block interface.

    This is the base block definition for a CenterPillar model.
    """

    def apply(x):
        input_depth = x.shape.as_list()[-1]
        stride = 2 if downsample else 1

        residual = x

        x = layers.Conv2D(
            filters,
            3,
            stride,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters,
            3,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        if downsample:
            residual = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(
                residual
            )

        if input_depth != filters:
            residual = layers.Conv2D(
                filters,
                1,
                1,
                padding="same",
                use_bias=False,
                kernel_initializer=initializers.VarianceScaling(),
                kernel_regularizer=regularizers.L2(l2=1e-4),
            )(residual)
            residual = layers.BatchNormalization()(residual)
            residual = layers.ReLU()(residual)

        x = x + residual

        return x

    return apply


def SkipBlock(filters):
    def apply(x):
        x = layers.Conv2D(
            filters,
            1,
            1,
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x

    return apply


def DownSampleBlock(filters, num_blocks):
    def apply(x):
        x = Block(filters, downsample=True)(x)

        for _ in range(num_blocks - 1):
            x = Block(filters, downsample=False)(x)

        return x

    return apply


def UpSampleBlock(filters):
    def apply(x, lateral_input):
        x = layers.Conv2DTranspose(
            filters,
            3,
            2,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        lateral_input = SkipBlock(filters)(lateral_input)

        x = x + lateral_input
        x = Block(filters, downsample=False)(x)

        return x

    return apply
