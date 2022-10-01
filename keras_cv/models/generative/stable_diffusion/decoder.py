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

import tensorflow as tf
from tensorflow import keras

from keras_cv.models.generative.stable_diffusion.__internal__.layers.attention_block import (
    AttentionBlock,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.group_normalization import (
    GroupNormalization,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.resnet_block import (
    ResnetBlock,
)


class Decoder(keras.Sequential):
    def __init__(self, img_height, img_width, name=None):
        super().__init__(
            [
                keras.layers.Input((img_height // 8, img_width // 8, 4)),
                keras.layers.Rescaling(1.0 / 0.18215),
                PaddedConv2D(4, 1),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(256),
                ResnetBlock(256),
                ResnetBlock(256),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                ResnetBlock(128),
                GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ],
            name=name,
        )
